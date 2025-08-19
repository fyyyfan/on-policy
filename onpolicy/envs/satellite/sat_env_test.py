import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

import numpy as np
from types import SimpleNamespace
from onpolicy.envs.satellite.scenarios.satellite_scenario import Scenario
from onpolicy.envs.satellite.core import SatelliteAction
import random

# python -m onpolicy.envs.satellite.sat_env_test
# 该脚本用于测试卫星环境的初始化和步进功能

# 策略选择：1-最近可见卫星，2-随机可见卫星（可能不迁移）
STRATEGY_TYPE = 2

def get_dummy_args():
    # 假设有2颗卫星、2个用户
    args = SimpleNamespace()
    args.episode_length = 20
    args.dt = 20.0 # 60
    args.num_users = 8
    # 卫星初始化参数
    args.num_sats = 6
    args.h = 7000 # km，轨道半径6788
    args.angle = 51.664  # 轨道倾角
    args.P_num = 1 #轨道面数
    # TLE数据
    # args.tle_list_line1 = f"1 44716U 19074D   25187.23278464  .00110409  00000+0  17717-2 0  9991"
    # args.tle_list_line2 = _generate_tles_line2(args.num_sats, args.h, args.angle, args.P_num)
    
    args.sat_comp_resource = [100] * args.num_sats  # 计算资源Gcycles/s
    args.sat_tran_power = [10] * args.num_sats
    args.sat_tran_gain = [36] * args.num_sats
    args.sat_rec_gain = [41] * args.num_sats
    # 用户初始化参数
    args.user_task_size = [np.random.randint(200, 500) for _ in range(args.num_users)]
    args.instance_size = [np.random.randint(500, 1000) for _ in range(args.num_users)]
    args.user_lon = [100.0, 110.0, 120.0, 130.0, 105.0, 115.0, 125.0, 135.0] # 经度[100.0, 120.0, 110.0, 115.0, 130.0]
    args.user_lat = [40.0, 50.0, 47.0, 45.0, 42.0, 35.0, 37.0, 46.0] # 纬度 [40.0, 50.0, 35.0, 42.0, 45.0]
    args.start_time = (2024, 1, 3, 8, 0, 0)
    return args


def _get_visible_neighbor_sats_for_user(agent, user, world):
    """返回用户对当前agent可迁移的可见邻居卫星及其距离数组[(sat, dist), ...]"""
    candidates = []
    
    # 策略1需要包含当前卫星自己
    # 检查当前卫星对用户的可见性
    current_sat_dist = world.user_sat_visibility.get((user.id, agent.id), -1)
    if current_sat_dist is not None and current_sat_dist > 0:
        candidates.append((agent, current_sat_dist))
    
    # 检查邻居卫星对用户的可见性
    for sat in agent.target_sat_list:
        dist = world.user_sat_visibility.get((user.id, sat.id), -1)
        if dist is not None and dist > 0:
            candidates.append((sat, dist))
    return candidates


def apply_strategy(strategy_type, world):
    """
    根据策略为所有agent生成动作：
    - 策略1：迁移到用户可见卫星中距离最近且在agent.target_sat_list内的卫星，或保持当前卫星
    - 策略2：在用户可见的邻居卫星中随机选择目标（也可能选择不迁移）
    """
    for agent in world.satellites:
        # 默认不迁移
        agent.action = SatelliteAction(0)

        # 没有服务实例或没有服务用户则不迁移
        if not agent.instance_list or not agent.service_users:
            continue

        # 选择一个用户驱动迁移（这里取第一个有对应实例的用户）
        chosen_user = None
        for user in agent.service_users:
            # 确保该用户的服务实例在本卫星上
            if any(ins.service_id == user.service_instance.service_id for ins in agent.instance_list):
                chosen_user = user
                break
        if chosen_user is None:
            continue

        # 候选目标卫星：当前卫星 + 与agent相邻且对用户可见的卫星
        candidates = _get_visible_neighbor_sats_for_user(agent, chosen_user, world)
        if not candidates:
            continue

        target_sat = None
        if strategy_type == 1:
            # 最近可见卫星（包括当前卫星自己）
            target_sat = min(candidates, key=lambda x: x[1])[0]
            # 如果选择的是当前卫星自己，则不迁移
            if target_sat.id == agent.id:
                target_sat = None
        elif strategy_type == 2:
            # 随机选择一个可见邻居，或不迁移
            # 以等概率从可见邻居+不迁移中选择
            pool = [c[0] for c in candidates] + [None]
            target_sat = random.choice(pool)
        else:
            # 未知策略，保持不迁移
            target_sat = None

        if target_sat is None:
            continue

        # service_id取所选用户的服务实例ID
        service_id = chosen_user.service_instance.service_id
        # 校验该实例确实在当前卫星的实例列表中；若意外缺失，退化为第一个实例
        if not any(ins.service_id == service_id for ins in agent.instance_list):
            service_id = agent.instance_list[0].service_id

        # 编码动作并下发
        action_id = agent.action.encode(service_id, target_sat.id, agent.target_sat_list)
        agent.action.action_id = action_id
        # 可选：打印调试
        print(f"[策略{strategy_type}] agent {agent.id}: user {chosen_user.id} -> target_sat {target_sat.id}, service_id {service_id}, action_id {action_id}")


def main():
    args = get_dummy_args()
    scenario = Scenario()
    world = scenario.make_world(args)
    print("World created.")
    print(f"Number of satellites: {len(world.satellites)}")
    print(f"Number of users: {len(world.user_clusters)}")
    for i, sat in enumerate(world.satellites):
        print(f"初始状态 Satellite {i}: comp_resource={sat.comp_resource}, instance_list={sat.instance_list}")
    for i, user in enumerate(world.user_clusters):
        print(f"初始状态 User {i}: current_sat={user.current_sat.id if user.current_sat else None}")
    # 绘制初始状态
    world.plot_step_positions_interactive(step_idx=0)

    # 测试环境步进
    print("\nStep the environment:")
    
    # 新增：回合延迟统计
    episode_delays = {
        'comm_delays': [],
        'comp_delays': [],
        'migration_delays': [],
        'total_delays': []
    }
    
    for step in range(args.episode_length):
        print(f"\n--- Step {step+1} ---")
        # # 手动给每个agent赋action
        # for agent in world.satellites:
        #     # 这里只是示例：每步都尝试迁移第0个服务到第1号卫星
        #     if agent.id == 0 and len(agent.target_sat_list) > 0:
        #         action_id = agent.action.encode(0, agent.target_sat_list[0].id, agent.target_sat_list)
        #         agent.action.action_id = action_id
        #     elif agent.id == 1 and len(agent.target_sat_list) > 0:
        #         action_id = agent.action.encode(0, agent.target_sat_list[0].id, agent.target_sat_list)
        #         agent.action.action_id = action_id
        #     elif agent.id == 2 and len(agent.target_sat_list) > 0:
        #         action_id = agent.action.encode(0, agent.target_sat_list[0].id, agent.target_sat_list)
        #         agent.action.action_id = action_id
        #     else:
        #         agent.action = SatelliteAction(0)
        # 使用策略生成动作（替代手动赋值）
        apply_strategy(STRATEGY_TYPE, world)
        world.step()
        
        # 新增：收集当前step的延迟数据
        step_delays = {
            'comm_delays': [],
            'comp_delays': [],
            'migration_delays': [],
            'total_delays': []
        }
        
        # 遍历所有用户，收集延迟数据
        for user in world.user_clusters:
            if user.current_sat is not None:
                # 计算通信延迟
                comm_delay = world._compute_communication_delay(user, user.current_sat)
                if comm_delay > 0 and not np.isnan(comm_delay) and not np.isinf(comm_delay):
                    step_delays['comm_delays'].append(comm_delay)
                
                # 计算计算延迟
                comp_delay = world._compute_computation_delay(user, user.current_sat)
                if comp_delay > 0 and not np.isnan(comp_delay) and not np.isinf(comp_delay):
                    step_delays['comp_delays'].append(comp_delay)
                
                # 获取迁移延迟
                migration_delay = user.migration_delay
                if migration_delay > 0 and not np.isnan(migration_delay) and not np.isinf(migration_delay):
                    step_delays['migration_delays'].append(migration_delay)
                
                # 计算总延迟
                total_delay = comm_delay + comp_delay + migration_delay
                if total_delay > 0 and not np.isnan(total_delay) and not np.isinf(total_delay):
                    step_delays['total_delays'].append(total_delay)
        
        # 将当前step的延迟数据添加到回合统计中
        for delay_type in step_delays:
            if step_delays[delay_type]:  # 如果有延迟数据
                episode_delays[delay_type].append(step_delays[delay_type])
        
        # 输出当前step的延迟统计
        print(f"=== Step {step+1} 延迟统计 ===")
        for delay_type, delays in step_delays.items():
            if delays:
                avg_delay = np.mean(delays)
                print(f"  {delay_type}: 平均延迟 = {avg_delay:.6f}s (用户数: {len(delays)})")
            else:
                print(f"  {delay_type}: 无有效延迟数据")
        
        # 计算reward
        for agent in world.satellites:
            reward_n = scenario.reward_agent(agent, world)
            print(f"计算卫星奖励 Satellite {agent.id} reward: {reward_n:.2f}")
        
        # world.plot_step_positions_interactive(step+1)
        # 打印状态
        for i, sat in enumerate(world.satellites):
            print(f"第{step+1}步",f"Satellite {i}: comp_resource={sat.comp_resource}, instance_list={[ins for ins in sat.instance_list]}")
        for i, user in enumerate(world.user_clusters):
            print(f"第{step+1}步",f"User {i}: current_sat={user.current_sat.id if user.current_sat else None}")
    
    # 新增：回合结束后的延迟统计汇总
    print("\n" + "="*50)
    print("回合延迟统计汇总")
    print("="*50)
    
    for delay_type, step_delays_list in episode_delays.items():
        if step_delays_list:  # 如果有延迟数据
            # 计算每个step的平均延迟
            step_averages = []
            for step_delays in step_delays_list:
                if step_delays:  # 确保该step有延迟数据
                    step_averages.append(np.mean(step_delays))
            
            if step_averages:
                # 计算整个回合的平均延迟
                episode_average = np.mean(step_averages)
                print(f"{delay_type}:")
                print(f"  回合平均延迟: {episode_average:.6f}s")
                # print(f"  有效步数: {len(step_averages)}/{len(episode_delays_list)}")
                # print(f"  各步平均延迟: {[f'{d:.6f}' for d in step_averages]}")
            else:
                print(f"{delay_type}: 无有效延迟数据")
        else:
            print(f"{delay_type}: 无延迟数据")
    
    print("="*50)

if __name__ == "__main__":
    main()