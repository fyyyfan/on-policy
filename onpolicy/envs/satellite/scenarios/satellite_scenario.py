import numpy as np
from core import SatelliteWorld, Satellite, UserCluster, ServiceInstance, Time, Walker
from mpe.scenario import BaseScenario

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        satellite_positions = []
        walker = Walker(args.num_sats, args.h, args.angle, args.P_num, 
                        args.sat_comp_resource, args.sat_tran_power, args.sat_tran_gain, args.sat_rec_gain)
        # TODO: 测试绘制卫星坐标
        # self._plot_satellite_positions(satellite_positions)

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
        重置world参数, 用于初始化卫星部署的服务、用户的当前卫星等状态
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


    def reward(self, agent, world):
        '''
        定义 agent 的奖励函数
        '''
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        '''
        定义 agent 的观测空间组成（输入策略网络）
        '''
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)    


    def _plot_satellite_positions(self, positions):
        # 提取 x, y, z 坐标
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]

        # 创建三维图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制散点图
        ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o')

        # 设置图形标题和轴标签
        ax.set_title("卫星三维坐标分布")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")
        ax.set_zlabel("Z 坐标")

        # 显示图形
        plt.savefig("satellite_positions.png")