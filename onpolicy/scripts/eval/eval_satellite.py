#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import csv
import time

import torch

from onpolicy.config import get_config
from onpolicy.envs.satellite.SAT_env import SATEnv
from onpolicy.envs.env_wrappers import ChooseSubprocVecEnv, ChooseDummyVecEnv


def make_train_env(all_args):
    """
    创建训练环境的工厂函数
    支持多线程并行环境创建
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Satellite":
                # 为每个线程创建独立的Satellite环境，使用不同的随机种子
                env = SATEnv(all_args, (all_args.seed + rank * 1000))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    # 根据线程数选择环境类型：单线程用DummyVecEnv，多线程用SubprocVecEnv
    if all_args.n_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """
    创建评估环境的工厂函数
    使用不同的随机种子确保评估的独立性
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Satellite":
                # 评估环境使用更大的种子偏移，确保与训练环境完全独立
                env = SATEnv(
                    all_args, (all_args.seed * 50000 + rank * 10000))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    # 根据评估线程数选择环境类型
    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """
    解析Satellite特定的命令行参数
    """
    # 卫星环境特有参数
    parser.add_argument('--scenario_name', type=str,
                        default='satellite_scenario', help="卫星场景名称")
    parser.add_argument('--start_time', type=str,
                        default='[2024,1,3,8,0,0]', help="开始时间")
    parser.add_argument('--dt', type=float,
                        default=60.0, help="物理世界时间步长(秒)")
    parser.add_argument('--num_users', type=int,
                        default=5, help="用户数量")
    parser.add_argument('--user_lon', type=str,
                        default='[100.0,120.0,140.0,160.0,180.0]', help="用户经度列表")
    parser.add_argument('--user_lat', type=str,
                        default='[40.0,45.0,50.0,35.0,30.0]', help="用户纬度列表")
    parser.add_argument('--num_sats', type=int,
                        default=6, help="卫星数量")
    parser.add_argument('--h', type=float,
                        default=7000.0, help="卫星轨道半径(km)")
    parser.add_argument('--angle', type=float,
                        default=0.0, help="卫星轨道倾角(度)")
    parser.add_argument('--P_num', type=int,
                        default=1, help="卫星轨道面数")
    parser.add_argument('--sat_comp_resource', type=str,
                        default='[100.0,100.0,100.0,100.0,100.0,100.0]', help="卫星计算资源")
    parser.add_argument('--sat_tran_power', type=str,
                        default='[10.0,10.0,10.0,10.0,10.0,10.0]', help="卫星传输功率")
    parser.add_argument('--sat_tran_gain', type=str,
                        default='[1.0,1.0,1.0,1.0,1.0,1.0]', help="卫星传输增益")
    parser.add_argument('--sat_rec_gain', type=str,
                        default='[1.0,1.0,1.0,1.0,1.0,1.0]', help="卫星接收增益")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def collect_delay_data(envs, episode, step, total_steps, n_rollout_threads):
    """
    收集当前时刻所有环境的延迟数据
    返回：平均服务延迟、平均迁移延迟
    """
    all_service_delays = []
    all_migration_delays = []
    
    try:
        # 遍历所有环境
        for env_id in range(n_rollout_threads):
            env = envs.envs[env_id]
            world = env.world
            
            # 遍历所有卫星和用户
            for sat in world.satellites:
                for user in sat.service_users:
                    # 获取服务状态
                    service_status_dict = world.get_user_service_status(sat)
                    service_status = service_status_dict.get(user.id, False)
                    
                    if service_status:
                        # 计算各种延迟组件
                        compute_delay = world._compute_computation_delay(user, sat) * 1000  # 转为ms
                        comm_delay = world._compute_communication_delay(user, sat) * 1000  # 转为ms
                        
                        # 服务延迟 = 计算延迟 + 通信延迟
                        service_delay = compute_delay + comm_delay
                        all_service_delays.append(service_delay)
                        
                        # 检查是否有迁移延迟
                        migration_delay = 0.0
                        if sat.action.target_sat and sat.action.service_instance and sat.action.user:
                            if sat.action.user.id == user.id:
                                migration_delay = world._compute_migration_delay(sat, user, sat.action.target_sat) * 1000
                                all_migration_delays.append(migration_delay)
        
        # 计算平均值
        avg_service_delay = np.mean(all_service_delays) if all_service_delays else 0.0
        avg_migration_delay = np.mean(all_migration_delays) if all_migration_delays else 0.0
        
        return avg_service_delay, avg_migration_delay, len(all_service_delays), len(all_migration_delays)
        
    except Exception as e:
        print(f"收集延迟数据时出错: {e}")
        return 0.0, 0.0, 0, 0


def main(args):
    # 获取基础配置
    parser = get_config()
    all_args = parse_args(args, parser)

    # 根据选择的算法自动配置相关参数
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        # R-MAPPO启用循环策略，适合处理部分可观察性
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        # MAPPO禁用循环策略，使用标准策略网络
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        # IPPO禁用中心化价值函数，每个智能体独立学习
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # 验证评估模式已启用
    assert all_args.use_eval, ("u need to set use_eval be True")
    # 验证模型目录已设置（用于加载训练好的模型）
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    

    # GPU/CPU设备配置
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            # 启用确定性计算，确保结果可重复
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 创建结果存储目录
    # 目录结构：results/Satellite/scenario_name/algorithm_name/experiment_name/
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Wandb实验跟踪配置
    if all_args.use_wandb:
        # 初始化wandb实验，记录配置和结果
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="evaluation",
                         reinit=True)
    else:
        # 本地实验管理：自动创建run1, run2等子目录
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # 设置进程标题，便于系统监控
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # 设置随机种子，确保实验可重复性
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 初始化环境
    envs = make_train_env(all_args)  # 训练环境（用于模型推理）
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None  # 评估环境
    num_agents = all_args.num_sats  # 卫星环境中智能体数量等于卫星数量

    # 构建配置字典，传递给Runner
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # 卫星环境只支持共享策略，统一从shared模块导入
    from onpolicy.runner.shared.satellite_runner import SatelliteRunner as Runner

    # 创建Runner实例并执行评估
    runner = Runner(config)
    
    # 执行评估，收集延迟数据
    print("开始执行Satellite环境评估...")
    
    # 评估配置
    n_eval_episodes = 10  # 评估的episode数量
    episode_length = all_args.episode_length
    
    # 延迟数据记录
    delay_records = []
    
    for episode in range(n_eval_episodes):
        print(f"评估Episode {episode + 1}/{n_eval_episodes}")
        
        # 重置环境
        obs, share_obs, available_actions = eval_envs.reset()
        
        # 初始化RNN状态（如果使用循环网络）
        if all_args.use_recurrent_policy:
            eval_rnn_states = np.zeros((all_args.n_eval_rollout_threads, *runner.buffer.rnn_states.shape[2:]), dtype=np.float32)
        else:
            eval_rnn_states = None
            
        eval_masks = np.ones((all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
        
        episode_service_delays = []
        episode_migration_delays = []
        
        for step in range(episode_length):
            # 准备策略网络
            runner.trainer.prep_rollout()
            
            # 获取动作
            if all_args.use_recurrent_policy:
                eval_action, eval_rnn_states = runner.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    available_actions=np.concatenate(available_actions) if available_actions is not None else None,
                    deterministic=True
                )
                eval_actions = np.array(np.split(runner._t2n(eval_action), all_args.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(runner._t2n(eval_rnn_states), all_args.n_eval_rollout_threads))
            else:
                eval_action = runner.trainer.policy.act(
                    np.concatenate(obs),
                    available_actions=np.concatenate(available_actions) if available_actions is not None else None,
                    deterministic=True
                )
                eval_actions = np.array(np.split(runner._t2n(eval_action), all_args.n_eval_rollout_threads))
            
            # 卫星环境的动作处理
            if eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError("Satellite environment only supports Discrete action space")

            # 执行环境步进
            obs, share_obs, rewards, dones, infos, available_actions = eval_envs.step(eval_actions_env)
            
            # 收集延迟数据（每5步收集一次，避免数据过多）
            if step % 5 == 0:
                avg_service_delay, avg_migration_delay, service_count, migration_count = collect_delay_data(
                    eval_envs, episode, step, episode * episode_length + step, all_args.n_eval_rollout_threads
                )
                
                if service_count > 0:
                    episode_service_delays.append(avg_service_delay)
                if migration_count > 0:
                    episode_migration_delays.append(avg_migration_delay)
                
                # 记录延迟数据
                delay_record = {
                    'episode': episode + 1,
                    'step': step + 1,
                    'avg_service_delay_ms': round(avg_service_delay, 3),
                    'avg_migration_delay_ms': round(avg_migration_delay, 3),
                    'service_count': service_count,
                    'migration_count': migration_count
                }
                delay_records.append(delay_record)
            
            # 更新RNN状态和掩码
            if all_args.use_recurrent_policy:
                eval_rnn_states[dones == True] = np.zeros(((dones == True).sum(), *runner.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
            eval_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        
        # Episode结束，计算平均延迟
        if episode_service_delays:
            episode_avg_service_delay = np.mean(episode_service_delays)
            print(f"Episode {episode + 1} 平均服务延迟: {episode_avg_service_delay:.3f} ms")
        else:
            episode_avg_service_delay = 0.0
            print(f"Episode {episode + 1} 无有效服务延迟数据")
            
        if episode_migration_delays:
            episode_avg_migration_delay = np.mean(episode_migration_delays)
            print(f"Episode {episode + 1} 平均迁移延迟: {episode_avg_migration_delay:.3f} ms")
        else:
            episode_avg_migration_delay = 0.0
            print(f"Episode {episode + 1} 无迁移延迟数据")
    
    # 计算总体评估结果
    if delay_records:
        all_service_delays = [record['avg_service_delay_ms'] for record in delay_records if record['service_count'] > 0]
        all_migration_delays = [record['avg_migration_delay_ms'] for record in delay_records if record['migration_count'] > 0]
        
        if all_service_delays:
            overall_avg_service_delay = np.mean(all_service_delays)
            print(f"\n=== 总体评估结果 ===")
            print(f"所有用户平均服务延迟: {overall_avg_service_delay:.3f} ms")
            print(f"服务延迟标准差: {np.std(all_service_delays):.3f} ms")
            print(f"服务延迟范围: [{np.min(all_service_delays):.3f}, {np.max(all_service_delays):.3f}] ms")
        
        if all_migration_delays:
            overall_avg_migration_delay = np.mean(all_migration_delays)
            print(f"所有用户平均迁移延迟: {overall_avg_migration_delay:.3f} ms")
            print(f"迁移延迟标准差: {np.std(all_migration_delays):.3f} ms")
            print(f"迁移延迟范围: [{np.min(all_migration_delays):.3f}, {np.max(all_migration_delays):.3f}] ms")
        
        # 保存延迟数据到CSV文件
        delay_csv_path = run_dir / "evaluation_delay_data.csv"
        with open(delay_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=delay_records[0].keys())
            writer.writeheader()
            writer.writerows(delay_records)
        print(f"延迟数据已保存到: {delay_csv_path}")
        
        # 如果使用wandb，记录评估结果
        if all_args.use_wandb:
            if all_service_delays:
                wandb.log({
                    "evaluation/overall_avg_service_delay_ms": overall_avg_service_delay,
                    "evaluation/service_delay_std_ms": np.std(all_service_delays),
                    "evaluation/service_delay_min_ms": np.min(all_service_delays),
                    "evaluation/service_delay_max_ms": np.max(all_service_delays)
                })
            if all_migration_delays:
                wandb.log({
                    "evaluation/overall_avg_migration_delay_ms": overall_avg_migration_delay,
                    "evaluation/migration_delay_std_ms": np.std(all_migration_delays),
                    "evaluation/migration_delay_min_ms": np.min(all_migration_delays),
                    "evaluation/migration_delay_max_ms": np.max(all_migration_delays)
                })
    
    # 清理资源
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # 完成实验记录
    if all_args.use_wandb:
        run.finish()
    else:
        print("评估完成，结果已保存到本地目录")


if __name__ == "__main__":
    main(sys.argv[1:])
