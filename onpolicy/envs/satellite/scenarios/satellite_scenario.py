import numpy as np
from onpolicy.envs.satellite.core import SatelliteWorld, Satellite, UserCluster, ServiceInstance, Time, Walker, SatelliteAction, SatelliteObs, logger
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
        logger.info(f"初始化时间: {t}")

        # 1. 定义卫星星座
        walker = Walker(args.num_sats, args.h, args.angle, args.P_num, 
                        args.sat_comp_resource, args.sat_tran_power, args.sat_tran_gain, args.sat_rec_gain)
        
        # 保存初始时间到walker对象中，供reset_world使用
        # walker.initial_time = t

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
        
        # 重置
        self.reset_world(world)
        # 打印初始分配状态
        print("==== 初始分配状态 ====")
        logger.info("==== 初始分配状态 ====")
        for user in world.user_clusters:
            if user.current_sat is not None:
                print(f"用户 {user.id} 分配给卫星 {user.current_sat.id}，服务实例ID: {user.service_instance.service_id}，实例大小: {user.service_instance.instance_size}，任务大小: {user.task_size}")
                logger.info(f"用户 {user.id} 分配给卫星 {user.current_sat.id}，服务实例ID: {user.service_instance.service_id}，实例大小: {user.service_instance.instance_size}，任务大小: {user.task_size}")
            else:
                print(f"用户 {user.id} 未分配到可见卫星，服务实例ID: {user.service_instance.service_id}，实例大小: {user.service_instance.instance_size}，任务大小: {user.task_size}")
                logger.info(f"用户 {user.id} 未分配到可见卫星，服务实例ID: {user.service_instance.service_id}，实例大小: {user.service_instance.instance_size}，任务大小: {user.task_size}")
        for sat in world.satellites:
            print(f"卫星 {sat.id} 剩余资源: {sat.comp_resource}，实例列表: {[ins.service_id for ins in sat.instance_list]}，服务用户: {[u.id for u in sat.service_users]}")
            logger.info(f"卫星 {sat.id} 剩余资源: {sat.comp_resource}，实例列表: {[ins.service_id for ins in sat.instance_list]}，服务用户: {[u.id for u in sat.service_users]}")
        logger.info("=====================")

        return world

    def reset_world(self, world):
        '''
        重置world参数, 用于每个episode初始化卫星部署的服务、用户的服务请求等状态
        '''
        # 1. 随机更新服务实例的大小
        for user in world.user_clusters:
            user.service_instance.instance_size = np.random.randint(500, 1000)

        # 2. 随机生成每个用户的地理位置
        # for user in world.user_clusters:
        #     user.lon = np.random.uniform(90, 180)  # 经度范围 
        #     user.lat = np.random.uniform(40, 60)  # 纬度范围 
        
        # 3. 随机生成每个用户的任务请求,用户和实例对象的关联关系不变
        for user in world.user_clusters:
            # 随机任务请求参数
            user.task_size = np.random.randint(100, 500)  # 比如任务大小 30~100
            
        # 4. 随机初始化每颗卫星的资源
        for sat in world.satellites:
            sat.comp_resource = np.random.randint(1000, 2000)
            sat.cpu_resource = np.random.randint(100, 200)  # CPU资源 (Gcycles/s)
            sat.instance_list = []
            sat.service_users = []  # 清空服务用户列表，避免重复
            sat.visible_user = []  # 清空可见用户列表
            sat.target_sat_list = []  # 清空目标卫星列表
            # 重置卫星动作和状态空间
            sat.action = SatelliteAction()  # 重置动作空间
            sat.state = SatelliteObs()  # 重置状态空间

        # 5. 清空用户当前卫星关联
        for user in world.user_clusters:
            user.current_sat = None  # 清空用户当前卫星

        # 6. 其它状态重置
        world.world_step = 0
        # 重置时间到初始状态 - 创建新的Time对象
        world.current_time = Time(
            year=world.initial_time.year,
            month=world.initial_time.month,
            day=world.initial_time.day,
            hour=world.initial_time.hour,
            minute=world.initial_time.minute,
            second=world.initial_time.second
        )
        logger.info(f"[重置] 时间已重置为: {world.current_time}")
        # 如果没有保存初始时间，则保持当前时间不变
        world.sat_topology = {}  # 清空卫星拓扑关系
        world.sat_links = {}  # 清空卫星间连接关系
        world.user_sat_visibility = {}  # 清空用户与卫星的可见性关系

        # 5. 重新更新链路状态和可见性矩阵
        world._update_link_states(world.current_time)
        world._update_visibility_matrix(world.current_time)

        # 8. 初始化用户与卫星的关联
        # 遍历每个用户，在所有可见卫星中随机分配一个
        for user in world.user_clusters:
            visible_sats = []
            # 遍历所有卫星，找到可见的卫星
            for sat in world.satellites:
                visibility_key = (user.id, sat.id)
                if visibility_key in world.user_sat_visibility:
                    distance = world.user_sat_visibility[visibility_key]
                    # 检查是否可见（距离 > 0）
                    if distance > 0:
                        visible_sats.append(sat)
            # 如果有可见卫星，随机选择一个
            if visible_sats:
                selected_sat = random.choice(visible_sats)
                selected_sat.instance_list.append(user.service_instance)
                selected_sat.service_users.append(user)
                user.current_sat = selected_sat
                distance = world.user_sat_visibility[(user.id, selected_sat.id)]
                print(f"[初始化] 用户{user.id}随机分配给卫星{selected_sat.id}，距离{distance:.2f}km")
                logger.info(f"[初始化] 用户{user.id}随机分配给卫星{selected_sat.id}，距离{distance:.2f}km")
            else:
                user.current_sat = None
                print(f"[初始化] 用户{user.id}没有找到可见卫星")
                logger.info(f"[初始化] 用户{user.id}没有找到可见卫星")
        # # 8. 初始化用户与卫星的关联
        # # 遍历每个用户，分配距离最近的可见卫星
        # for user in world.user_clusters:
        #     min_distance = float('inf')
        #     nearest_sat = None
            
        #     # 遍历所有卫星，找到距离最近的可见卫星
        #     for sat in world.satellites:
        #         visibility_key = (user.id, sat.id)
        #         if visibility_key in world.user_sat_visibility:
        #             distance = world.user_sat_visibility[visibility_key]
        #             # 检查是否可见（距离 > 0）且距离更近
        #             if distance > 0 and distance < min_distance:
        #                 min_distance = distance
        #                 nearest_sat = sat
            
            # # 如果找到可见卫星，则分配
            # if nearest_sat is not None:
            #     nearest_sat.instance_list.append(user.service_instance)
            #     nearest_sat.service_users.append(user)
            #     user.current_sat = nearest_sat
            #     print(f"[初始化] 用户{user.id}分配给距离最近的卫星{nearest_sat.id}，距离{min_distance:.2f}km")
            #     logger.info(f"[初始化] 用户{user.id}分配给距离最近的卫星{nearest_sat.id}，距离{min_distance:.2f}km")
            # else:
            #     user.current_sat = None
            #     print(f"[初始化] 用户{user.id}没有找到可见卫星")
            #     logger.info(f"[初始化] 用户{user.id}没有找到可见卫星")

    def reward_agent(self, agent, world: SatelliteWorld):
        '''
        定义单个智能体的奖励函数
        sat环境真正用到的只有单个智能体的奖励函数，在environment.py中计算共享奖励
        
        惩罚机制设计说明：
        1. 服务失败惩罚（-800.0）：当用户服务完全失败时的严重惩罚，优先级最高
        2. 可见时间惩罚（0~-200.0）：当剩余可见时间<30s时的渐进式惩罚，鼓励提前迁移
           - 惩罚权重为200.0，小于服务失败惩罚，避免过度惩罚
           - 当T_rem=0时（已不可见），服务失败惩罚会生效，可见时间惩罚为0
           - 当T_rem接近30s时，惩罚接近0，平滑过渡
        3. 关系：可见时间惩罚是服务失败惩罚的"预警机制"，帮助智能体在服务完全失败前主动迁移
        '''
        baseline_delay = 1200
        delay_weight = 1
        # # 计算延迟奖励-ms
        # total_delay = world._calculate_total_delay(agent) * 1000
        # delay_reward = delay_weight * (baseline_delay-total_delay)
        
        # 计算服务失败惩罚
        service_failure_penalty = 500.0  # 服务失败惩罚权重
        
        # 新增：剩余可见时间惩罚参数
        visibility_time_threshold = 40.0  # 剩余可见时间阈值（秒）
        visibility_penalty_weight = 300.0  # 可见时间惩罚权重
        
        # 初始化奖励
        total_reward = 0.0
        service_count = 0
        
        # 获取卫星上所有用户的服务状态
        service_status_dict = world.get_user_service_status(agent)
        
        # 对卫星服务的每个用户分别计算奖励
        for user in agent.service_users:
            service_count += 1
            
            # 从服务状态字典中获取该用户的服务状态
            service_status = service_status_dict.get(user.id, False)
            
            # 如果服务失败，直接进行惩罚，跳过延迟计算
            if not service_status:
                user_reward = -service_failure_penalty  # 直接惩罚
                print(f"[奖励计算] 卫星{agent.id}的用户{user.id}服务失败，直接惩罚: {user_reward:.2f}")
                logger.info(f"[奖励计算] 卫星{agent.id}的用户{user.id}服务失败，直接惩罚: {user_reward:.2f}")
            else:
                # 服务成功，计算延迟奖励
                user_delay = world._calculate_user_delay(user, agent) * 1000  # 转换为ms
                user_delay_reward = delay_weight * (baseline_delay - user_delay)
                
                # 新增：计算剩余可见时间惩罚
                T_rem, T_vis = world.compute_remaining_visibility_time(agent, user, world.current_time)
                visibility_penalty = 0.0
                
                if T_rem < visibility_time_threshold and T_rem > 0:
                    # 根据剩余可见时间按权重惩罚，剩余时间越少惩罚越重
                    penalty_ratio = (visibility_time_threshold - T_rem) / visibility_time_threshold
                    visibility_penalty = visibility_penalty_weight * penalty_ratio
                    print(f"[奖励计算] 卫星{agent.id}的用户{user.id}剩余可见时间{T_rem:.2f}s < {visibility_time_threshold}s，惩罚: {visibility_penalty:.2f}")
                    logger.info(f"[奖励计算] 卫星{agent.id}的用户{user.id}剩余可见时间{T_rem:.2f}s < {visibility_time_threshold}s，惩罚: {visibility_penalty:.2f}")
                
                # 用户总奖励 = 延迟奖励 - 可见时间惩罚
                user_reward = user_delay_reward - visibility_penalty
                print(f"[奖励计算] 卫星{agent.id}的用户{user.id}服务成功，延迟={user_delay:.2f}ms, 延迟奖励={user_delay_reward:.2f}, 可见时间惩罚={visibility_penalty:.2f}, 用户奖励={user_reward:.2f}")
                logger.info(f"[奖励计算] 卫星{agent.id}的用户{user.id}服务成功，延迟={user_delay:.2f}ms, 延迟奖励={user_delay_reward:.2f}, 可见时间惩罚={visibility_penalty:.2f}, 用户奖励={user_reward:.2f}")
            
            # 累加用户奖励
            total_reward += user_reward
        
        # # 总奖励 = 延迟奖励 - 服务失败惩罚
        # total_reward = delay_reward - service_failure_weight * service_failure_penalty
        
        # print(f"[奖励计算] 卫星{agent.id}: 延迟奖励={delay_reward:.6f}, 服务失败惩罚={service_failure_penalty:.6f}, 总奖励={total_reward:.6f}")
        # logger.info(f"[奖励计算] 卫星{agent.id}: 延迟奖励={delay_reward:.6f}, 服务失败惩罚={service_failure_penalty:.6f}, 总奖励={total_reward:.6f}")
            
            # print(f"[奖励计算] 卫星{agent.id}的用户{user.id}: 延迟={user_delay:.2f}ms, 延迟奖励={user_delay_reward:.2f}, 服务失败惩罚={service_failure_penalty:.2f}, 用户奖励={user_reward:.2f}")
            # logger.info(f"[奖励计算] 卫星{agent.id}的用户{user.id}: 延迟={user_delay:.2f}ms, 延迟奖励={user_delay_reward:.2f}, 服务失败惩罚={service_failure_penalty:.2f}, 用户奖励={user_reward:.2f}")
        
        # 如果卫星没有服务任何用户，给予基础奖励
        if service_count == 0:
            final_reward = 0.0
            print(f"[奖励计算] 卫星{agent.id}没有服务任何用户，奖励为0")
            logger.info(f"[奖励计算] 卫星{agent.id}没有服务任何用户，奖励为0")
        else:
            # 计算平均奖励，确保公平性
            avg_reward = total_reward / service_count
            final_reward = avg_reward
            print(f"[奖励计算] 卫星{agent.id}: 服务{service_count}个用户, 平均奖励={final_reward:.2f}")
            logger.info(f"[奖励计算] 卫星{agent.id}: 服务{service_count}个用户, 平均奖励={final_reward:.2f}")
        
        # 观察 avg_reward 的大致范围，比如它在 [-1000, 800] 之间
        # 选择一个合适的缩放因子，例如 100
        REWARD_SCALING_FACTOR = 100.0 
        scaled_reward = final_reward / REWARD_SCALING_FACTOR
        print(f"[奖励计算] 卫星{agent.id}的奖励缩放后为{scaled_reward:.2f}")
        logger.info(f"[奖励计算] 卫星{agent.id}的奖励缩放后为{scaled_reward:.2f}")

        return scaled_reward
    

    def reward(self, agent, world):
        '''
        定义所有 agent 的全局奖励函数
        '''
        # 智能体的奖励取决于所有服务的总延迟
        total_delay = 0
        for sat in world.satellites:
            total_delay += world._calculate_total_delay(sat)
            # print(f"卫星{sat.id}的总延迟: {total_delay}")
        
        # 获取基准延迟 baseline_delay = world.get_baseline_delay()
        baseline_delay = 1.2
        
        # 使用基准延迟计算相对奖励，避免持续负值
        reward = baseline_delay - total_delay
        
        # 更新当前episode的奖励记录
        # world.update_episode_reward(reward)
        
        # print(f"[DEBUG] 奖励计算: 基准延迟={baseline_delay:.6f}, 总延迟={total_delay:.6f}, 奖励={reward:.6f}")
        
        return reward

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
