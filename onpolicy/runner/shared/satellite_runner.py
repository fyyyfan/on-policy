import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import os
import logging
import csv
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

def _t2n(x):
    return x.detach().cpu().numpy()

class SatelliteRunner(Runner):
    """Runner class to perform training, evaluation, and data collection for the Satellite environment. 
    See parent class for details."""
    
    def __init__(self, config):
        super(SatelliteRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # # 可视化配置
        # viz_save_interval = 50  # 每50步保存一次可视化
        # viz_enabled = hasattr(self.all_args, 'enable_visualization') and self.all_args.enable_visualization

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # # 记录episode级别的统计信息
            # episode_rewards = []
            # migration_stats = {'no_migration': 0, 'migration': 0}

            episode_infos = []
            # 新增：收集延迟数据
            episode_delays = []
            # 新增：收集服务失败数据
            episode_service_failures = []
            # 新增：收集详细延迟组件数据
            episode_comm_delays = []
            episode_comp_delays = []
            episode_migration_delays = []
            
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                # # 记录迁移统计
                # for action in actions_env:
                #     if action == 0:
                #         migration_stats['no_migration'] += 1
                #     else:
                #         migration_stats['migration'] += 1
                    
                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
                episode_infos.append(infos)
                
                # 新增：收集延迟数据
                step_delays = []
                # 新增：收集服务失败数据
                step_service_failures = []
                # 新增：收集详细延迟组件数据
                step_comm_delays = []
                step_comp_delays = []
                step_migration_delays = []
                
                # 从infos中获取延迟和服务失败信息
                for info in infos:
                    env_delays = []
                    env_service_failures = 0
                    # 新增：环境级别的详细延迟组件
                    env_comm_delays = []
                    env_comp_delays = []
                    env_migration_delays = []
                    
                    # 遍历每个智能体的信息
                    for agent_id in range(len(info)):
                        agent_info = info[agent_id]
                        # 检查是否有延迟数据，如果没有则跳过
                        if 'delay_data' in agent_info:
                            delay_data = agent_info['delay_data']
                            for user_key, user_data in delay_data.items():
                                if user_data.get('service_status', False):
                                    total_delay = user_data.get('total_delay_s', 0.0)
                                    if total_delay > 0 and not np.isnan(total_delay) and not np.isinf(total_delay):
                                        env_delays.append(total_delay)
                                    
                                    # 新增：收集详细延迟组件
                                    comm_delay = user_data.get('comm_delay_s', 0.0)
                                    comp_delay = user_data.get('comp_delay_s', 0.0)
                                    migration_delay = user_data.get('migration_delay_s', 0.0)
                                    
                                    if comm_delay > 0 and not np.isnan(comm_delay) and not np.isinf(comm_delay):
                                        env_comm_delays.append(comm_delay)
                                    if comp_delay > 0 and not np.isnan(comp_delay) and not np.isinf(comp_delay):
                                        env_comp_delays.append(comp_delay)
                                    if migration_delay > 0 and not np.isnan(migration_delay) and not np.isinf(migration_delay):
                                        env_migration_delays.append(migration_delay)
                                else:
                                    # 服务失败，计数加1
                                    env_service_failures += 1
                    
                    if env_delays:  # 如果有延迟数据
                        step_delays.append(np.mean(env_delays))
                    step_service_failures.append(env_service_failures)
                    
                    # 新增：记录环境级别的详细延迟组件
                    if env_comm_delays:
                        step_comm_delays.append(np.mean(env_comm_delays))
                    if env_comp_delays:
                        step_comp_delays.append(np.mean(env_comp_delays))
                    if env_migration_delays:
                        step_migration_delays.append(np.mean(env_migration_delays))
                
                if step_delays:  # 如果该步有延迟数据
                    episode_delays.append(step_delays)
                if step_service_failures:  # 如果该步有服务失败数据
                    episode_service_failures.append(step_service_failures)
                
                # 新增：记录步骤级别的详细延迟组件
                if step_comm_delays:
                    episode_comm_delays.append(step_comm_delays)
                if step_comp_delays:
                    episode_comp_delays.append(step_comp_delays)
                if step_migration_delays:
                    episode_migration_delays.append(step_migration_delays)
                
                # # 记录奖励
                # episode_rewards.append(np.sum(rewards))

                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions

                # insert data into buffer
                self.insert(data)
                
                # # 训练过程中的可视化
                # if viz_enabled:
                #     total_num_steps = episode * self.episode_length * self.n_rollout_threads + step * self.n_rollout_threads
                #     self.visualize_training_progress(episode, step, total_num_steps, viz_save_interval)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # # 创建episode摘要可视化
            # if viz_enabled and episode % 10 == 0:  # 每10个episode创建一次摘要
            #     self.create_training_summary_visualization(episode, episode_rewards, migration_stats)
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                # 卫星环境特定的日志信息
                if self.env_name == "Satellite":
                    env_infos = {}
                    # 记录每个卫星的个体奖励
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for step_info in episode_infos:
                            for info in step_info:
                                if 'individual_reward' in info[agent_id].keys():
                                    idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'satellite%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews
                    
                    # 新增：记录用户总体平均延迟
                    if episode_delays:
                        # 计算每个step中所有并行环境的平均延迟
                        step_average_delays = []
                        for step_delay_list in episode_delays:
                            if step_delay_list:  # 确保有数据
                                step_average_delays.append(np.mean(step_delay_list))
                        
                        # 计算整个episode的平均延迟
                        if step_average_delays:
                            episode_average_delay = np.mean(step_average_delays)
                            env_infos['users_average_delay'] = [episode_average_delay]  # 包装成列表
                            # logger.info(f"==== Episode {episode} 用户平均延迟: {episode_average_delay:.6f}s")
                            # logger.info(f"==== Episode {episode} 延迟数据统计: {len(episode_delays)} steps, {len(step_average_delays)} valid steps")
                    else:
                        logger.warning(f"Episode {episode} 没有收集到延迟数据")
                    
                    # 新增：记录服务失败次数
                    if episode_service_failures:
                        # 计算每个step中所有并行环境的服务失败总次数
                        step_total_failures = []
                        for step_failure_list in episode_service_failures:
                            if step_failure_list:  # 确保有数据
                                step_total_failures.append(sum(step_failure_list))
                        
                        # 计算整个episode的服务失败总次数
                        if step_total_failures:
                            episode_total_failures = sum(step_total_failures)
                            episode_average_failures = np.mean(step_total_failures)
                            env_infos['service_failure_total_count'] = [episode_total_failures]
                            env_infos['service_failure_average_per_step'] = [episode_average_failures]  # 包装成列表
                            logger.info(f"===== Episode {episode} 服务失败总次数: {episode_total_failures}")
                            logger.info(f"===== Episode {episode} 每步平均服务失败次数: {episode_average_failures:.2f}")
                            # logger.info(f"Episode {episode} 服务失败数据统计: {len(episode_service_failures)} steps, {len(step_total_failures)} valid steps")
                    else:
                        logger.warning(f"Episode {episode} 没有收集到服务失败数据")
                    
                    # 新增：记录详细延迟组件统计
                    # 通信延迟统计
                    if episode_comm_delays:
                        step_average_comm_delays = []
                        for step_comm_delay_list in episode_comm_delays:
                            if step_comm_delay_list:
                                step_average_comm_delays.append(np.mean(step_comm_delay_list))
                        
                        if step_average_comm_delays:
                            episode_average_comm_delay = np.mean(step_average_comm_delays)
                            env_infos['users_average_comm_delay'] = [episode_average_comm_delay]
                            logger.info(f"==== Episode {episode} 用户平均通信延迟: {episode_average_comm_delay:.6f}s")
                    
                    # 计算延迟统计
                    if episode_comp_delays:
                        step_average_comp_delays = []
                        for step_comp_delay_list in episode_comp_delays:
                            if step_comp_delay_list:
                                step_average_comp_delays.append(np.mean(step_comp_delay_list))
                        
                        if step_average_comp_delays:
                            episode_average_comp_delay = np.mean(step_average_comp_delays)
                            env_infos['users_average_comp_delay'] = [episode_average_comp_delay]
                            logger.info(f"==== Episode {episode} 用户平均计算延迟: {episode_average_comp_delay:.6f}s")
                    
                    # 迁移延迟统计
                    if episode_migration_delays:
                        step_average_migration_delays = []
                        for step_migration_delay_list in episode_migration_delays:
                            if step_migration_delay_list:
                                step_average_migration_delays.append(np.mean(step_migration_delay_list))
                        
                        if step_average_migration_delays:
                            episode_average_migration_delay = np.mean(step_average_migration_delays)
                            env_infos['users_average_migration_delay'] = [episode_average_migration_delay]
                            logger.info(f"==== Episode {episode} 用户平均迁移延迟: {episode_average_migration_delay:.6f}s")
                    
                    # # 记录迁移统计信息
                    # env_infos['migration_stats'] = migration_stats
                    # env_infos['episode_total_reward'] = np.sum(episode_rewards)
                    
                    # 记录卫星环境特定的指标
                    # for info in infos:
                    #     if 'satellite_metrics' in info[0].keys():
                    #         metrics = info[0]['satellite_metrics']
                    #         for key, value in metrics.items():
                    #             if key not in env_infos:
                    #                 env_infos[key] = []
                    #             env_infos[key].append(value)

                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length 
                print("[debug] ===== 平均轮次奖励 is {} =====".format(train_infos["average_episode_rewards"]), "step", total_num_steps)
                logger.info("[debug] ===== 平均轮次奖励 is {} =====".format(train_infos["average_episode_rewards"]), "step", total_num_steps)
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        if self.buffer.available_actions is not None:
            self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        # ====== DEBUG: 检查观测和动作掩码 ======
        obs = self.buffer.obs[step]
        share_obs = self.buffer.share_obs[step]
        available_actions = self.buffer.available_actions[step] if self.buffer.available_actions is not None else None
        print(f"====== collect step={step} ======")
        print(f"[DEBUG] obs shape: {np.shape(obs)} share_obs shape: {np.shape(share_obs)}")
        if np.isnan(obs).any() or np.isinf(obs).any():
            print("[ERROR] obs 存在 nan 或 inf！")
            print(obs)
        if np.isnan(share_obs).any() or np.isinf(share_obs).any():
            print("[ERROR] share_obs 存在 nan 或 inf！")
            print(share_obs)
        if available_actions is not None:
            print(f"[DEBUG] available_actions shape: {np.shape(available_actions)}")
            if np.isnan(available_actions).any() or np.isinf(available_actions).any():
                print("[ERROR] available_actions 存在 nan 或 inf！")
                print(available_actions)
            if (available_actions.sum(axis=-1) == 0).any():
                print("[ERROR] 有agent的动作掩码全为0！")
                print(available_actions)
        # ====== END DEBUG ======
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        # 卫星环境的动作处理 - 使用Discrete动作空间
        if self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError("Satellite environment only supports Discrete action space")

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                available_actions=np.concatenate(eval_available_actions) if eval_available_actions is not None else None,
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # 卫星环境的评估动作处理
            if self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError("Satellite environment only supports Discrete action space")

            # Observe reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of satellite: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the satellite environment."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs, share_obs, available_actions = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    available_actions=np.concatenate(available_actions) if available_actions is not None else None,
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                # 卫星环境的渲染动作处理
                if envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError("Satellite environment only supports Discrete action space")

                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    # def visualize_training_progress(self, episode, step, total_steps, save_interval=100):
    #     """
    #     在训练过程中阶段性可视化卫星环境和服务部署状态
        
    #     Args:
    #         episode: 当前episode编号
    #         step: 当前step编号
    #         total_steps: 总步数
    #         save_interval: 保存间隔（每隔多少步保存一次可视化）
    #     """
    #     # 只在特定间隔保存可视化
    #     if step % save_interval == 0:
    #         try:
    #             # 创建可视化目录
    #             viz_dir = f"training_visualization/episode_{episode:04d}"
    #             os.makedirs(viz_dir, exist_ok=True)
                
    #             # 为每个环境生成可视化
    #             for env_id in range(self.n_rollout_threads):
    #                 env = self.envs.envs[env_id]
                    
    #                 # 生成HTML交互式可视化
    #                 html_filename = f"{viz_dir}/step_{step:04d}_env_{env_id}.html"
    #                 env.render(mode='html', save_dir=html_filename)
                    
    #                 # 生成PNG静态图片
    #                 png_filename = f"{viz_dir}/step_{step:04d}_env_{env_id}.png"
    #                 env.render(mode='png', save_dir=png_filename)
                    
    #                 # 记录当前状态信息
    #                 status_info = {
    #                     'episode': episode,
    #                     'step': step,
    #                     'total_steps': total_steps,
    #                     'env_id': env_id,
    #                     'world_step': env.world.world_step,
    #                     'current_time': str(env.world.current_time),
    #                     'satellites': [
    #                         {
    #                             'id': sat.id,
    #                             'comp_resource': sat.comp_resource,
    #                             'service_users': [u.id for u in sat.service_users],
    #                             'instance_list': [ins.service_id for ins in sat.instance_list]
    #                         }
    #                         for sat in env.world.satellites
    #                     ],
    #                     'users': [
    #                         {
    #                             'id': user.id,
    #                             'current_sat': user.current_sat.id if user.current_sat else None,
    #                             'service_status': env.world.get_user_service_status(user)
    #                         }
    #                         for user in env.world.user_clusters
    #                     ]
    #                 }
                    
    #                 # 保存状态信息到JSON文件
    #                 import json
    #                 status_filename = f"{viz_dir}/step_{step:04d}_env_{env_id}_status.json"
    #                 with open(status_filename, 'w', encoding='utf-8') as f:
    #                     json.dump(status_info, f, indent=2, ensure_ascii=False)
                
    #             logger.info(f"训练可视化已保存: episode={episode}, step={step}, 目录={viz_dir}")
                
    #         except Exception as e:
    #             logger.error(f"保存训练可视化时出错: {e}")

    # def create_training_summary_visualization(self, episode, total_rewards, migration_stats):
    #     """
    #     创建训练摘要可视化，展示整个episode的迁移策略和性能
        
    #     Args:
    #         episode: episode编号
    #         total_rewards: 总奖励列表
    #         migration_stats: 迁移统计信息
    #     """
    #     try:
    #         import matplotlib.pyplot as plt
    #         import seaborn as sns
            
    #         # 创建摘要目录
    #         summary_dir = f"training_summary/episode_{episode:04d}"
    #         os.makedirs(summary_dir, exist_ok=True)
            
    #         # 1. 奖励曲线图
    #         plt.figure(figsize=(12, 8))
            
    #         plt.subplot(2, 2, 1)
    #         plt.plot(total_rewards)
    #         plt.title(f'Episode {episode} 奖励曲线')
    #         plt.xlabel('Step')
    #         plt.ylabel('Total Reward')
    #         plt.grid(True)
            
    #         # 2. 迁移统计图
    #         plt.subplot(2, 2, 2)
    #         if migration_stats:
    #             migration_counts = list(migration_stats.values())
    #             migration_labels = list(migration_stats.keys())
    #             plt.bar(migration_labels, migration_counts)
    #             plt.title('迁移动作统计')
    #             plt.xlabel('迁移类型')
    #             plt.ylabel('次数')
    #             plt.xticks(rotation=45)
            
    #         # 3. 服务成功率
    #         plt.subplot(2, 2, 3)
    #         # 这里可以添加服务成功率的统计
            
    #         # 4. 资源利用率
    #         plt.subplot(2, 2, 4)
    #         # 这里可以添加资源利用率的统计
            
    #         plt.tight_layout()
    #         plt.savefig(f"{summary_dir}/episode_{episode:04d}_summary.png", dpi=300, bbox_inches='tight')
    #         plt.close()
            
    #         logger.info(f"训练摘要可视化已保存: {summary_dir}")
            
    #     except Exception as e:
    #         logger.error(f"创建训练摘要可视化时出错: {e}")
