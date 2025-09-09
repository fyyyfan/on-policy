#!/usr/bin/env python
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "."))
sys.path.insert(0, project_root)

import numpy as np
from onpolicy.config import get_config
from onpolicy.envs.satellite.SAT_env import SATEnv

def test_env_creation():
    """测试环境创建"""
    
    # 创建参数解析器
    parser = get_config()
    
    # 设置最小化的调试参数
    debug_args = [
        # 基础参数
        "--algorithm_name", "mappo",
        "--experiment_name", "debug_test",
        "--seed", "1",
        "--cuda", "False",
        "--n_training_threads", "1",
        "--n_rollout_threads", "1",
        "--n_eval_rollout_threads", "1",
        "--num_env_steps", "100",
        "--use_wandb", "False",
        "--wandb_name", "debug_test",
        "--user_name", "debug_user",
        
        # 环境参数
        "--env_name", "Satellite",
        "--scenario_name", "satellite_scenario",
        "--episode_length", "20",
        
        # 卫星环境特有参数
        "--start_time", "2024", "1", "3", "8", "0", "0",
        "--dt", "60.0",
        "--num_users", "2",
        "--user_lon", "100.0", "120.0",
        "--user_lat", "40.0", "45.0",
        "--num_sats", "3",
        "--h", "7000.0",
        "--angle", "0.0",
        "--P_num", "1",
        "--sat_comp_resource", "100.0", "100.0", "100.0",
        "--sat_tran_power", "10.0", "10.0", "10.0",
        "--sat_tran_gain", "1.0", "1.0", "1.0",
        "--sat_rec_gain", "1.0", "1.0", "1.0",
        
        # 网络参数
        "--share_policy", "True",
        "--use_centralized_V", "True",
        "--hidden_size", "64",
        "--layer_N", "1",
        "--use_ReLU", "True",
        "--use_valuenorm", "True",
        "--use_feature_normalization", "True",
        "--use_orthogonal", "True",
        "--gain", "0.01",
        
        # 循环网络参数
        "--use_recurrent_policy", "False",
        "--recurrent_N", "1",
        "--data_chunk_length", "10",
        
        # 优化器参数
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--opti_eps", "1e-5",
        "--weight_decay", "0",
        
        # PPO参数
        "--ppo_epoch", "15",
        "--use_clipped_value_loss", "True",
        "--clip_param", "0.2",
        "--num_mini_batch", "1",
        "--entropy_coef", "0.01",
        "--use_max_grad_norm", "True",
        "--max_grad_norm", "10.0",
        "--use_gae", "True",
        "--gamma", "0.99",
        "--gae_lambda", "0.95",
        "--use_proper_time_limits", "False",
        "--use_huber_loss", "True",
        "--use_value_active_masks", "True",
        "--use_policy_active_masks", "True",
        "--huber_delta", "10.0",
        
        # 运行参数
        "--use_linear_lr_decay", "False",
        
        # 保存和日志参数
        "--save_interval", "1",
        "--log_interval", "5",
        
        # 评估参数
        "--use_eval", "False",
        "--eval_interval", "25",
        "--eval_episodes", "32",
        
        # 渲染参数
        "--save_gifs", "False",
        "--use_render", "False",
        "--render_episodes", "5",
        "--ifi", "0.1",
        
        # 预训练参数
        "--model_dir", "None",
    ]
    
    all_args = parser.parse_known_args(debug_args)[0]
    all_args.env_name = "Satellite"
    
    print("=== 开始测试环境创建 ===")
    print(f"卫星数量: {all_args.num_sats}")
    print(f"用户数量: {all_args.num_users}")
    print(f"Episode长度: {all_args.episode_length}")
    
    try:
        # 创建环境
        print("正在创建卫星环境...")
        env = SATEnv(all_args)
        print("✓ 环境创建成功")
        
        # 测试环境属性
        print(f"智能体数量: {env.n}")
        print(f"观测空间数量: {len(env.observation_space)}")
        print(f"共享观测空间数量: {len(env.share_observation_space)}")
        print(f"动作空间数量: {len(env.action_space)}")
        
        # 测试重置
        print("正在测试环境重置...")
        obs_n, share_obs_n, available_actions = env.reset()
        print("✓ 环境重置成功")
        print(f"观测形状: {[obs.shape for obs in obs_n]}")
        print(f"共享观测形状: {[share_obs.shape for share_obs in share_obs_n]}")
        print(f"动作掩码形状: {available_actions.shape}")
        
        # 测试步进
        print("正在测试环境步进...")
        # 随机动作
        actions = [np.random.randint(0, env.action_space[i].n) for i in range(env.n)]
        obs_n, share_obs_n, reward_n, done_n, info_n, available_actions = env.step(actions)
        print("✓ 环境步进成功")
        print(f"奖励: {reward_n}")
        print(f"完成状态: {done_n}")
        
        # 测试多步
        print("正在测试多步运行...")
        for step in range(5):
            actions = [np.random.randint(0, env.action_space[i].n) for i in range(env.n)]
            obs_n, share_obs_n, reward_n, done_n, info_n, available_actions = env.step(actions)
            print(f"步骤 {step+1}: 奖励={reward_n}, 完成={done_n}")
            if any(done_n):
                print("Episode结束，重置环境")
                obs_n, share_obs_n, available_actions = env.reset()
        
        print("✓ 所有测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env_creation() 