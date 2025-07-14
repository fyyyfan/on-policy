import numpy as np
from core import SatelliteWorld, Satellite, UserCluster, ServiceInstance, Time, Walker
from mpe.scenario import BaseScenario



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
        

        # 2. 创建卫星世界
        world = SatelliteWorld(walker, t)
        world.satellites = walker.create_satellites() #创建卫星对象
        for sat in world.satellites:
            # 打印卫星信息
            print(sat.sat)
            pos_xyz = sat._satellite_pos(Time(*args.start_time))
            print("卫星", sat.id, "的xyz位置", pos_xyz)
            print("卫星", sat.id, "的星下点位置", sat._satellite_pos(Time(*args.start_time), 'spt'))
        world.world_length = args.episode_length
        world.dt = args.dt if hasattr(args, 'dt') else 1.0

        # 3. 创建用户簇和服务实例
        num_users = args.num_users
        for i in range(num_users):
            # 生成服务实例
            service_instance = ServiceInstance(
                id=i,
                size=args.instance_size[i]
            )
            # 生成用户簇
            user = UserCluster(
                id=i,
                lon=args.user_lon[i],
                lat=args.user_lat[i],
                service_instance=service_instance,
                task_size=args.user_task_size[i]
            )
            world.user_clusters.append(user)
        
        # 4. 初始化卫星间连接关系
        world._update_link_states(t)

        # 5. 初始化用户与卫星的可见性关系
        world._update_visibility_matrix(t)

        # 6. 初始化卫星状态（分配初始服务实例/用户）
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
            # 打印初始状态
            print(f"Initial State: User {user.id} assigned to Satellite {user.current_sat.id if user.current_sat else 'None'}")

        # make initial conditions
        # self.reset_world(world)

        return world

    def reset_world(self, world):
        '''
        重置world参数, 用于每个episode初始化卫星部署的服务、用户的当前卫星等状态
        '''
        # # random properties for agents
        # world.assign_agent_colors()

        # world.assign_landmark_colors()

        # # set random initial states
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)

    def reward_agent(self, agent, world):
        '''
        定义单个智能体的奖励函数
        '''
        return world._calculate_total_delay(agent)
    

    def reward(self, agent, world):
        '''
        定义所有 agent 的全局奖励函数
        '''
        # 智能体的奖励取决于所有服务的总延迟
        total_delay = 0
        for sat in world.satellites:
            total_delay += world._calculate_total_delay(sat)
            # print(f"卫星{sat.id}的总延迟: {total_delay}")
        return total_delay

    def observation(self, agent, world):
        '''
        返回所有智能体的观测
        '''
        obs_n = []
        for sat in self.satellites:
            obs = self.observation_agent(sat)
            obs_n.append(obs)
        return np.array(obs_n, dtype=np.float32)
            

    def observation_agent(self, sat: Satellite):
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
                dist = self.sat_links.get((sat.id, neighbor.id), {}).get("distance", 0.0)
                obs.append(dist / MAX_DISTANCE)
            else:
                obs.append(0.0)

        # 6. 与可迁移卫星的链路传输速率（最多4个，归一化）
        for i in range(4):
            if i < len(sat.target_sat_list):
                neighbor = sat.target_sat_list[i]
                rate = self.sat_links.get((sat.id, neighbor.id), {}).get("data_rate", 0.0)
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
        # 例如打印本agent当前剩余资源和延迟
        info = {
            "agent_id": agent.id,
            "time": world.current_time,
            "service_instance": agent.instance_list,
            "service_users": agent.service_users,
            "action": agent.action
            # 也可以加任何你关心的其他指标
        }
        return info
