#!/usr/bin/env python3
"""
测试卫星环境可视化功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from onpolicy.envs.satellite.core import logger, Time, SatelliteWorld, Walker
from onpolicy.envs.satellite.scenarios.satellite_scenario import Scenario
from onpolicy.envs.satellite.environment import MultiAgentEnv

def test_visualization():
    """测试可视化功能"""
    
    # 创建简单的参数
    class Args:
        def __init__(self):
            self.episode_length = 10
            self.dt = 1.0
            self.num_sats = 3
            self.h = 550
            self.angle = 53
            self.P_num = 1
            self.sat_comp_resource = [1000, 1000, 1000]
            self.sat_tran_power = [1, 1, 1]
            self.sat_tran_gain = [10, 10, 10]
            self.sat_rec_gain = [41, 41, 41]
            self.num_users = 2
            self.start_time = [2025, 7, 6, 12, 0, 0]
            self.user_lon = [120, 130]
            self.user_lat = [30, 40]
    
    args = Args()
    
    try:
        # 创建场景和环境
        scenario = Scenario()
        world = scenario.make_world(args)
        
        env = MultiAgentEnv(
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward_agent,
            observation_callback=scenario.observation_agent,
            info_callback=scenario.info,
            done_callback=None
        )
        
        print("=== 测试不同渲染模式 ===")
        
        # 测试HTML模式
        print("1. 测试HTML交互式渲染...")
        env.render(mode='html', save_dir='test_html')
        print("   ✓ HTML渲染完成")
        
        # 测试PNG模式
        print("2. 测试PNG静态渲染...")
        env.render(mode='png', save_dir='test_png')
        print("   ✓ PNG渲染完成")
        
        # 测试RGB数组模式
        print("3. 测试RGB数组渲染...")
        rgb_array = env.render(mode='rgb_array')
        if rgb_array is not None:
            print(f"   ✓ RGB数组渲染完成，形状: {rgb_array.shape}")
        else:
            print("   ✗ RGB数组渲染失败")
        
        # 测试训练过程可视化
        print("\n=== 测试训练过程可视化 ===")
        
        # 模拟几个训练步骤
        for step in range(5):
            print(f"步骤 {step + 1}:")
            
            # 重置环境
            if step == 0:
                obs_n, share_obs_n, available_actions = env.reset()
            
            # 执行动作（简单的随机动作）
            import numpy as np
            actions = [np.random.randint(0, 5) for _ in range(env.n)]
            obs_n, share_obs_n, rewards, dones, infos, available_actions = env.step(actions)
            
            print(f"  动作: {actions}")
            print(f"  奖励: {rewards}")
            
            # 每2步渲染一次
            if step % 2 == 0:
                env.render(mode='html', save_dir=f'test_training/step_{step}')
                print(f"  ✓ 可视化已保存到 test_training/step_{step}")
        
        print("\n=== 可视化测试完成 ===")
        print("生成的文件:")
        print("- test_html/: HTML交互式可视化")
        print("- test_png/: PNG静态图片")
        print("- test_training/: 训练过程可视化")
        
    except Exception as e:
        logger.error(f"可视化测试过程中出现错误: {e}")
        print(f"错误: {e}")

def test_runner_visualization():
    """测试Runner中的可视化功能"""
    print("\n=== 测试Runner可视化功能 ===")
    
    # 这里可以添加Runner可视化功能的测试
    # 由于Runner需要完整的训练配置，这里只是示例
    print("Runner可视化功能需要在完整训练环境中测试")
    print("可以通过设置 --enable_visualization 参数来启用")

if __name__ == "__main__":
    test_visualization()
    test_runner_visualization() 