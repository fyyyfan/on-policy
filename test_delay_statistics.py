#!/usr/bin/env python3
"""
测试脚本：验证用户延迟统计功能
"""

import numpy as np

def test_delay_statistics():
    """测试延迟统计逻辑"""
    print("测试用户延迟统计功能...")
    
    # 模拟一个episode的延迟数据
    episode_delays = [
        [0.001, 0.002, 0.003],  # step 0: 3个环境的延迟
        [0.002, 0.003, 0.004],  # step 1: 3个环境的延迟
        [0.003, 0.004, 0.005],  # step 2: 3个环境的延迟
    ]
    
    print(f"原始延迟数据: {episode_delays}")
    
    # 计算每个step中所有并行环境的平均延迟
    step_average_delays = []
    for step_delay_list in episode_delays:
        if step_delay_list:  # 确保有数据
            step_average_delays.append(np.mean(step_delay_list))
    
    print(f"每步平均延迟: {step_average_delays}")
    
    # 计算整个episode的平均延迟
    if step_average_delays:
        episode_average_delay = np.mean(step_average_delays)
        print(f"Episode平均延迟: {episode_average_delay:.6f}s")
        print(f"延迟数据统计: {len(episode_delays)} steps, {len(step_average_delays)} valid steps")
    
    # 测试边界情况
    print("\n测试边界情况:")
    
    # 空数据
    empty_delays = []
    if empty_delays:
        print("空数据测试: 有数据")
    else:
        print("空数据测试: 无数据")
    
    # 包含无效数据
    invalid_delays = [
        [0.001, np.nan, 0.003],
        [0.002, np.inf, 0.004],
        [0.003, -0.001, 0.005],
    ]
    
    print(f"包含无效数据的原始数据: {invalid_delays}")
    
    # 过滤无效数据
    filtered_delays = []
    for step_delay_list in invalid_delays:
        valid_delays = [d for d in step_delay_list if d > 0 and not np.isnan(d) and not np.isinf(d)]
        if valid_delays:
            filtered_delays.append(valid_delays)
    
    print(f"过滤后的有效数据: {filtered_delays}")
    
    if filtered_delays:
        step_averages = [np.mean(step) for step in filtered_delays]
        episode_average = np.mean(step_averages)
        print(f"过滤后的Episode平均延迟: {episode_average:.6f}s")

if __name__ == "__main__":
    test_delay_statistics() 