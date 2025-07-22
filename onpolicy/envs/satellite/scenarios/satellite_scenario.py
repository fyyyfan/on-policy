import numpy as np
from onpolicy.envs.satellite.core import SatelliteWorld, Satellite, UserCluster, ServiceInstance, Time, Walker
from onpolicy.envs.mpe.scenario import BaseScenario
import random


class Scenario(BaseScenario):
    def make_world(self, args):
        '''
        Args:
            episode_length
            dt 物理世界时间步长
            num_sats
            tle_list
            num_users
            start_time
        '''
        # 这里假设初始时刻为 args.start_time
        t = Time(*args.start_time)
        print(f"初始化时间: {t}")

        # 1. 定义卫星星座
        walker = Walker(args.num_sats, args.h, args.angle, args.P_num, 
                        args.sat_comp_resource, args.sat_tran_power, args.sat_tran_gain, args.sat_rec_gain)
        
        # 保存初始时间到walker对象中，供reset_world使用
        walker.initial_time = t

        # 2. 创建卫星世界
        world = SatelliteWorld(walker, t)
        world.satellites = walker.create_satellites() #创建卫星对象
        # 设置智能体数量
        world.num_agents = args.num_sats
        for sat in world.satellites:
            # 打印卫星信息
            print(sat.sat)
            pos_xyz = sat._satellite_pos(Time(*args.start_time))
            print("卫星", sat.id, "的xyz位置", pos_xyz)
            print("卫星", sat.id, "的星下点位置", sat._satellite_pos(Time(*args.start_time), 'spt'))
        world.world_length = args.episode_length
        world.dt = args.dt if hasattr(args, 'dt') else 1.0

        # 3. 创建用户簇和服务实例,一对一
        num_users = args.num_users
        for i in range(num_users):
            # 生成服务实例
            service_instance = ServiceInstance(
                id=i,
                # size=args.instance_size[i]
                size=10  # 默认值，会在reset_world中随机更新
            )
            # 生成用户簇
            user = UserCluster(
                id=i,
                lon=args.user_lon[i] if hasattr(args, 'user_lon') and i < len(args.user_lon) else 0.0,
                lat=args.user_lat[i] if hasattr(args, 'user_lat') and i < len(args.user_lat) else 0.0,
                service_instance=service_instance,
                task_size=5  # 默认值，会在reset_world中随机更新
            )
            world.user_clusters.append(user)
        
        # 注意：不在这里初始化卫星状态和用户分配，这些会在reset_world中处理
        # 因为reset_world会随机初始化卫星资源和重新分配用户
        # # 4. 初始化卫星间连接关系
        # world._update_link_states(t)

        # # 5. 初始化用户与卫星的可见性关系
        # world._update_visibility_matrix(t)

        # # 6. 初始化卫星状态（分配初始服务实例/用户）
        # # 遍历每个用户，分配一个可见卫星
        # for user in world.user_clusters:
        #     assigned = False
        #     for sat in world.satellites:
        #         # 检查可见性，按id顺序遍历，如果卫星可见就分配给用户
        #         if world.user_sat_visibility.get((user.id, sat.id)) > 0:
        #             sat.instance_list.append(user.service_instance)
        #             sat.service_users.append(user)
        #             user.current_sat = sat
        #             assigned = True
        #             break
        #     # 若没有可见卫星，可根据需求处理（如随机分配或置None）
        #     if not assigned:
        #         user.current_sat = None
        #     # 打印初始状态
        #     print(f"Initial State: User {user.id} assigned to Satellite {user.current_sat.id if user.current_sat else 'None'}")

        # 重置
        self.reset_world(world)
        # 打印初始分配状态
        print("==== 初始分配状态 ====")
        for user in world.user_clusters:
            if user.current_sat is not None:
                print(f"用户 {user.id} 分配给卫星 {user.current_sat.id}，服务实例ID: {user.service_instance.service_id}，实例大小: {user.service_instance.instance_size}，任务大小: {user.task_size}")
            else:
                print(f"用户 {user.id} 未分配到可见卫星，服务实例ID: {user.service_instance.service_id}，实例大小: {user.service_instance.instance_size}，任务大小: {user.task_size}")
        for sat in world.satellites:
            print(f"卫星 {sat.id} 剩余资源: {sat.comp_resource}，实例列表: {[ins.service_id for ins in sat.instance_list]}，服务用户: {[u.id for u in sat.service_users]}")
        print("=====================")

        return world

    def reset_world(self, world):
        '''
        重置world参数, 用于每个episode初始化卫星部署的服务、用户的服务请求等状态
        '''
        # 1. 随机更新服务实例的大小
        for user in world.user_clusters:
            user.service_instance.instance_size = np.random.randint(10, 50)

        # 2. 随机生成每个用户的地理位置
        # for user in world.user_clusters:
        #     user.lon = np.random.uniform(90, 180)  # 经度范围 
        #     user.lat = np.random.uniform(40, 60)  # 纬度范围 
        
        # 3. 随机生成每个用户的任务请求,用户和实例对象的关联关系不变
        for user in world.user_clusters:
            # 随机任务请求参数
            user.task_size = np.random.randint(5, 30)  # 比如任务大小 5~30
            
        # 4. 随机初始化每颗卫星的资源
        for sat in world.satellites:
            sat.comp_resource = np.random.randint(30, 100)
            sat.instance_list = []

        # 5. 其它状态重置
        world.world_step = 0
        # 重置时间到初始状态（从world的walker中获取初始时间）
        if hasattr(world, 'walker') and hasattr(world.walker, 'initial_time'):
            world.current_time = world.walker.initial_time
        # 如果没有保存初始时间，则保持当前时间不变
        world.sat_topology = {}  # 清空卫星拓扑关系
        world.sat_links = {}  # 清空卫星间连接关系
        world.user_sat_visibility = {}  # 清空用户与卫星的可见性关系

        # 5. 重新更新链路状态和可见性矩阵
        world._update_link_states(world.current_time)
        world._update_visibility_matrix(world.current_time)

        # 6. 初始化用户与卫星的关联
        # 遍历每个用户，分配一个可见卫星
        for user in world.user_clusters:
            assigned = False
            for sat in world.satellites:
                # 检查可见性，按id顺序遍历，如果卫星可见就分配给用户
                if world.user_sat_visibility.get((user.id, sat.id)) > 0:
                    sat.instance_list.append(user.service_instance)
                    sat.service_users.append(user)
                    user.current_sat = sat
                    assigned = True
                    break
            # 若没有可见卫星，可根据需求处理（如随机分配或置None）
            if not assigned:
                user.current_sat = None
        # 获取所有卫星id
        all_sat_ids = [sat.id for sat in world.satellites]
        # 排除当前已分配的卫星
        if user.current_sat is not None:
            other_sat_ids = [sid for sid in all_sat_ids if sid != user.current_sat.id]
        else:
            other_sat_ids = all_sat_ids
        if other_sat_ids:
            random_sat_id = random.choice(other_sat_ids)
            random_sat = next(sat for sat in world.satellites if sat.id == random_sat_id)
            # 避免重复添加
            if user.service_instance not in random_sat.instance_list:
                random_sat.instance_list.append(user.service_instance)
            if user not in random_sat.service_users:
                random_sat.service_users.append(user)

    def reward_agent(self, agent, world):
        '''
        定义单个智能体的奖励函数
        '''
        return - world._calculate_total_delay(agent)
    

    def reward(self, agent, world):
        '''
        定义所有 agent 的全局奖励函数
        '''
        # 智能体的奖励取决于所有服务的总延迟
        total_delay = 0
        for sat in world.satellites:
            total_delay += world._calculate_total_delay(sat)
            # print(f"卫星{sat.id}的总延迟: {total_delay}")
        return - total_delay

    def observation(self, agent, world):
        '''
        返回所有智能体的观测
        '''
        obs_n = []
        for sat in world.satellites:
            obs = self.observation_agent(sat, world)
            obs_n.append(obs)
        return np.array(obs_n, dtype=np.float32)
            

    def observation_agent(self, sat: Satellite, world: SatelliteWorld):
        '''
        定义 agent 的观测空间组成（输入策略网络）
        '''
        obs = []
        
        # 设定归一化上限值（需根据环境实际情况合理设定）
        MAX_RESOURCE = 100.0
        MAX_INSTANCE_SIZE = 50.0
        MAX_DISTANCE = 3000.0  # 假设为卫星最大通信距离 km
        MAX_RATE = 100e6  # 假设为最大链路速率 100 Mbps
        
        # 1. 卫星剩余资源（归一化）
        obs.append(sat.comp_resource / MAX_RESOURCE)
        
        # 2. 当前部署的服务实例大小（最多2个，归一化）
        for i in range(2):
            if i < len(sat.instance_list):
                obs.append(sat.instance_list[i].instance_size / MAX_INSTANCE_SIZE)
            else:
                obs.append(0.0)
        
        # 3. 可见用户id（最多2个）
        for i in range(2):
            if i < len(sat.visible_user):
                obs.append(sat.visible_user[i].id)
            else:
                obs.append(0.0)
        
        # 4. 可见用户的服务请求实例大小（最多2个，归一化）
        for i in range(2):
            if i < len(sat.visible_user):
                user = sat.visible_user[i]
                if hasattr(user, 'service_instance') and hasattr(user.service_instance, 'instance_size'):
                    obs.append(user.service_instance.instance_size / MAX_INSTANCE_SIZE)
                else:
                    obs.append(0.0)
            else:
                obs.append(0.0)

        # 5. 与可迁移卫星的距离（最多4个，归一化）
        for i in range(4):
            if i < len(sat.target_sat_list):
                neighbor = sat.target_sat_list[i]
                dist = world.sat_links.get((sat.id, neighbor.id), {}).get("distance", 0.0)
                obs.append(dist / MAX_DISTANCE)
            else:
                obs.append(0.0)

        # 6. 与可迁移卫星的链路传输速率（最多4个，归一化）
        for i in range(4):
            if i < len(sat.target_sat_list):
                neighbor = sat.target_sat_list[i]
                rate = world.sat_links.get((sat.id, neighbor.id), {}).get("data_rate", 0.0)
                obs.append(rate / MAX_RATE)
            else:
                obs.append(0.0)

        # 7. 可迁移卫星的剩余资源（最多4个，归一化）
        for i in range(4):
            if i < len(sat.target_sat_list):
                neighbor = sat.target_sat_list[i]
                obs.append(neighbor.comp_resource / MAX_RESOURCE)
            else:
                obs.append(0.0)

        return np.array(obs, dtype=np.float32)


    def info(self, agent: Satellite, world: SatelliteWorld):
        # 目前没用上，例如打印本agent当前剩余资源和延迟
        info = {
            "agent_id": agent.id,
            "time": world.current_time,
            "service_instance": agent.instance_list,
            "service_users": agent.service_users,
            "action": agent.action
        
            # 也可以加任何你关心的其他指标
        }
        return info
