import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import numpy as np
from types import SimpleNamespace
from scenarios.satellite_scenario import Scenario
from core import SatelliteAction

# python -m onpolicy.envs.satellite.sat_env_test

def get_dummy_args():
    # 假设有2颗卫星、2个用户
    args = SimpleNamespace()
    args.episode_length = 10
    args.dt = 1.0
    args.num_users = 2
    # 卫星初始化参数
    args.num_sats = 12
    args.h = 7000 # km，轨道半径6788
    args.angle = 51.664  # 轨道倾角
    args.P_num = 2 #轨道面数
    # TLE数据
    # args.tle_list_line1 = f"1 44716U 19074D   25187.23278464  .00110409  00000+0  17717-2 0  9991"
    # args.tle_list_line2 = _generate_tles_line2(args.num_sats, args.h, args.angle, args.P_num)
    
    args.sat_comp_resource = [100] * args.num_sats  # 计算资源Gcycles/s
    args.sat_tran_power = [10] * args.num_sats
    args.sat_tran_gain = [1] * args.num_sats
    args.sat_rec_gain = [1] * args.num_sats
    # 用户初始化参数
    args.user_task_size = [5, 10]  # Mbit
    args.instance_size = [10, 20]
    args.user_lon = [100.0, 120.0] # 经度
    args.user_lat = [40.0, 50.0] # 纬度
    args.user_min_elev = [116, 120] # 最小仰角
    args.start_time = (2024, 1, 3, 8, 0, 0)
    return args

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

    # 测试环境步进
    print("\nStep the environment:")
    for step in range(3):
        print(f"\n--- Step {step+1} ---")
        # 手动给每个agent赋action
        for agent in world.satellites:
            # 这里只是示例：每步都尝试迁移第0个服务到第1号卫星
            if agent.id == 0 and len(agent.target_sat_list) > 0:
                action_id = agent.action.encode(0, agent.target_sat_list[0].id, agent.target_sat_list)
                agent.action = SatelliteAction(action_id)
            else:
                agent.action = SatelliteAction(0)
        world.step()
        # 打印状态
        for i, sat in enumerate(world.satellites):
            print(f"第{step+1}步",f"Satellite {i}: comp_resource={sat.comp_resource}, instance_list={[ins for ins in sat.instance_list]}")
        for i, user in enumerate(world.user_clusters):
            print(f"第{step+1}步",f"User {i}: current_sat={user.current_sat.id if user.current_sat else None}")

if __name__ == "__main__":
    main()