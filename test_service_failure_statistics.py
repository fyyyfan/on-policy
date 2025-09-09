#!/usr/bin/env python3
"""
测试脚本：验证服务失败统计功能
"""

import numpy as np

def test_service_failure_statistics():
    """测试服务失败统计逻辑"""
    print("测试服务失败统计功能...")
    
    # 模拟一个episode的服务失败数据
    episode_service_failures = [
        [2, 1, 3],  # step 0: 3个环境的服务失败次数
        [1, 2, 1],  # step 1: 3个环境的服务失败次数
        [3, 1, 2],  # step 2: 3个环境的服务失败次数
    ]
    
    print(f"原始服务失败数据: {episode_service_failures}")
    
    # 计算每个step中所有并行环境的服务失败总次数
    step_total_failures = []
    for step_failure_list in episode_service_failures:
        if step_failure_list:  # 确保有数据
            step_total_failures.append(sum(step_failure_list))
    
    print(f"每步服务失败总次数: {step_total_failures}")
    
    # 计算整个episode的服务失败总次数和平均值
    if step_total_failures:
        episode_total_failures = sum(step_total_failures)
        episode_average_failures = np.mean(step_total_failures)
        print(f"Episode服务失败总次数: {episode_total_failures}")
        print(f"Episode每步平均服务失败次数: {episode_average_failures:.2f}")
        print(f"服务失败数据统计: {len(episode_service_failures)} steps, {len(step_total_failures)} valid steps")
    
    # 测试边界情况
    print("\n测试边界情况:")
    
    # 空数据
    empty_failures = []
    if empty_failures:
        print("空数据测试: 有数据")
    else:
        print("空数据测试: 无数据")
    
    # 包含零值的数据
    zero_failures = [
        [0, 0, 0],  # 所有环境都没有服务失败
        [1, 0, 2],  # 部分环境有服务失败
        [0, 0, 0],  # 所有环境都没有服务失败
    ]
    
    print(f"包含零值的原始数据: {zero_failures}")
    
    # 计算服务失败统计
    step_totals = []
    for step_failure_list in zero_failures:
        if step_failure_list:  # 确保有数据
            step_totals.append(sum(step_failure_list))
    
    if step_totals:
        episode_total = sum(step_totals)
        episode_average = np.mean(step_totals)
        print(f"包含零值数据的Episode服务失败总次数: {episode_total}")
        print(f"包含零值数据的Episode每步平均服务失败次数: {episode_average:.2f}")
    
    # 模拟真实场景：不同step可能有不同的环境数量
    variable_env_failures = [
        [2, 1],      # step 0: 2个环境
        [1, 2, 3],   # step 1: 3个环境
        [0],          # step 2: 1个环境
    ]
    
    print(f"\n不同环境数量的原始数据: {variable_env_failures}")
    
    # 计算服务失败统计
    step_totals_var = []
    for step_failure_list in variable_env_failures:
        if step_failure_list:  # 确保有数据
            step_totals_var.append(sum(step_failure_list))
    
    if step_totals_var:
        episode_total_var = sum(step_totals_var)
        episode_average_var = np.mean(step_totals_var)
        print(f"不同环境数量的Episode服务失败总次数: {episode_total_var}")
        print(f"不同环境数量的Episode每步平均服务失败次数: {episode_average_var:.2f}")

if __name__ == "__main__":
    test_service_failure_statistics() 