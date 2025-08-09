import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from .core import SatelliteWorld, SatelliteAction, logger, predict_next_step_visibility, Time


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):

    def __init__(self, world: SatelliteWorld, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=True):
        '''
        卫星环境初始化
        Args:
            world: SatelliteWorld 对象
            reset_callback: 重置回调函数
            reward_callback: 奖励回调函数
            observation_callback: 观测回调函数
            info_callback: 信息回调函数
            done_callback: 结束回调函数
            post_step_callback: 后步回调函数-目前没有使用
            shared_viewer: 共享渲染器
            discrete_action: 离散动作
        '''
        # 1. 卫星环境的 world 应为 SatelliteWorld
        self.world = world
        self.world_length = self.world.world_length # world初始化时设定的最大时间步

        # 2. agents 指向 satellites
        self.agents = self.world.satellites
        self.n = self.world.num_agents

        # 3. 回调函数
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.post_step_callback = post_step_callback

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = discrete_action

        # # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        # self.discrete_action_input = False
        # self.force_discrete_action = False

        # 每个智能体共享奖励 if true, every agent has the same reward
        self.shared_reward = True
        #self.shared_reward = False


        # 4. 配置动作空间和观测空间——所有智能体组成的
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0

        for agent in self.agents:
            # 动作空间的大小取决于智能体可以迁移的服务数量和可见的目标卫星数量。
            # 为了定义一个固定的Gym空间，我们通常使用一个场景下的最大可能值。

            # 4.1 动作空间（假设为 Discrete，具体可根据你的动作定义调整）
            action_space_dim = 1 + len(self.world.user_clusters) * 4
            self.action_space.append(spaces.Discrete(action_space_dim))
            # 4.2 观测空间
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
            
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # # 5. 渲染相关（可选）
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()


    def get_available_actions(self):
        """
        为每个智能体生成动作掩码，确保只有合法的动作可以被选择。
        动作空间维度固定为 1 + 用户数 * max_target_sat_num。
        
        动作掩码为1的条件（同时满足以下4个要求）：
        1. 卫星instance_list当中包含该服务实例
        2. target_sat在agent.target_sat_list当中  
        3. target_sat有足够的资源满足迁移
        4. 请求服务实例instance的用户在target_sat.visible_user列表当中
        
        return: 
        available_actions: 每个智能体的动作掩码，形状为 (n, action_space_dim)
        """
        max_target_sat_num = 4  # 固定最大目标卫星数
        num_services = len(self.world.user_clusters)
        action_space_dim = 1 + num_services * max_target_sat_num
        available_actions = []

        for agent in self.agents:
            # 初始化为全0，维度为 action_space_dim
            agent_available_actions = np.zeros(action_space_dim, dtype=np.float32)
            # 动作0（不迁移）总是可用的
            agent_available_actions[0] = 1.0

            # 检查每个可能的迁移动作
            for service_id in range(num_services):
                # 要求1: 检查当前卫星是否有这个服务实例
                service_instance = None
                service_size = 0
                for instance in agent.instance_list:
                    if instance.service_id == service_id:
                        service_instance = instance
                        service_size = instance.instance_size
                        break
                
                if service_instance is None:
                    # 不满足要求1，跳过该服务
                    continue
                
                # 找到请求该服务的用户
                service_user = None
                for user in self.world.user_clusters:
                    if user.service_instance.service_id == service_id:
                        service_user = user
                        break
                
                if service_user is None:
                    # 找不到请求该服务的用户，跳过
                    continue
                
                # 遍历最大目标卫星数
                for target_idx in range(max_target_sat_num):
                    # 要求2: 检查target_idx是否在agent.target_sat_list范围内
                    if target_idx >= len(agent.target_sat_list):
                        # 不满足要求2，该动作掩码保持为0
                        continue
                    
                    target_sat = agent.target_sat_list[target_idx]
                    
                    # 要求3: 检查目标卫星是否有足够的计算资源
                    if target_sat.comp_resource < service_size:
                        # 不满足要求3，跳过该目标卫星
                        continue
                    
                    # print(f"服务实例{service_id} 用户{service_user.id}{service_user} 卫星{target_sat} 可见用户列表{target_sat.visible_user}")
                    # 要求4: 检查请求该服务的用户是否在目标卫星的可见用户列表中
                    if service_user not in target_sat.visible_user:
                        # 不满足要求4，跳过该目标卫星
                        continue
                    
                    # 所有要求都满足，设置动作掩码为1
                    action_id = 1 + service_id * max_target_sat_num + target_idx
                    agent_available_actions[action_id] = 1.0
                    print(f"[DEBUG] 卫星{agent.id}合法动作：迁移服务{service_id}->卫星{target_sat.id}")
            
            available_actions.append(agent_available_actions)
        return np.array(available_actions, dtype=np.float32)
    
    def get_available_actions_with_emergency(self):
        """
        为每个智能体生成动作掩码，确保只有合法的动作可以被选择。
        动作空间维度固定为 1 + 用户数 * max_target_sat_num。
        
        动作掩码为1的条件（同时满足以下4个要求）：
        1. 卫星instance_list当中包含该服务实例
        2. target_sat在agent.target_sat_list当中  
        3. target_sat有足够的资源满足迁移
        4. 请求服务实例instance的用户在target_sat.visible_user列表当中
        
        紧急迁移机制：
        以用户为单位，记录下一时刻不可见的卫星。当某个卫星在下一时刻不可见时，
        只允许迁移该卫星上的服务，其他服务的迁移动作设为0；
        如果没有可选的迁移动作，允许不迁移作为合法动作兜底
        
        return: 
        available_actions: 每个智能体的动作掩码，形状为 (n, action_space_dim)
        """
        max_target_sat_num = 4  # 固定最大目标卫星数
        num_services = len(self.world.user_clusters)
        action_space_dim = 1 + num_services * max_target_sat_num
        available_actions = []
        
        # 预测下一时刻的时间
        next_time = Time(
            year=self.world.current_time.year,
            month=self.world.current_time.month,
            day=self.world.current_time.day,
            hour=self.world.current_time.hour,
            minute=self.world.current_time.minute,
            second=self.world.current_time.second + self.world.dt
        )
        
        # 处理时间进位
        if next_time.second >= 60:
            next_time.minute += int(next_time.second // 60)
            next_time.second = next_time.second % 60
        if next_time.minute >= 60:
            next_time.hour += int(next_time.minute // 60)
            next_time.minute = next_time.minute % 60
        if next_time.hour >= 24:
            next_time.day += int(next_time.hour // 24)
            next_time.hour = next_time.hour % 24

        for agent in self.agents:
            # 初始化为全0，维度为 action_space_dim
            agent_available_actions = np.zeros(action_space_dim, dtype=np.float32)
            
            # 检查当前卫星是否有紧急情况：是否有用户在下一时刻不可见
            has_emergency = False
            emergency_users = []
            
            # 检查每个用户对当前卫星的下一时刻可见性
            for user in agent.service_users:
                # 预测下一时刻用户对当前卫星的可见性
                next_visibility = predict_next_step_visibility(user, agent, next_time)
                if not next_visibility:
                    has_emergency = True
                    emergency_users.append(user)
                    print(f"[紧急迁移] 卫星{agent.id}的用户{user.id}在下一时刻将不可见，需要紧急迁移")
            
            # 检查每个可能的迁移动作
            for service_id in range(num_services):
                # 要求1: 检查当前卫星是否有这个服务实例
                service_instance = None
                service_size = 0
                for instance in agent.instance_list:
                    if instance.service_id == service_id:
                        service_instance = instance
                        service_size = instance.instance_size
                        break
                
                if service_instance is None:
                    # 不满足要求1，跳过该服务
                    continue
                
                # 找到请求该服务的用户
                service_user = None
                for user in self.world.user_clusters:
                    if user.service_instance.service_id == service_id:
                        service_user = user
                        break
                
                if service_user is None:
                    # 找不到请求该服务的用户，跳过
                    continue
                
                # 如果有紧急情况，只允许迁移当前卫星上的服务
                if has_emergency:
                    # 检查当前服务是否在当前卫星上
                    service_on_current_sat = service_instance in agent.instance_list
                    if not service_on_current_sat:
                        # 如果服务不在当前卫星上，跳过该服务的迁移动作
                        # print(f"[紧急迁移] 卫星{agent.id}有紧急情况，跳过非当前卫星的服务{service_id}迁移")
                        logger.info(f"[紧急迁移] 卫星{agent.id}有紧急情况，跳过非当前卫星的服务{service_id}迁移")
                        continue
                    
                    # 检查该服务对应的用户是否在紧急用户列表中
                    if service_user not in emergency_users:
                        # 如果该服务对应的用户不在紧急用户列表中，跳过该服务的迁移动作
                        # print(f"[紧急迁移] 卫星{agent.id}有紧急情况，跳过非紧急用户{service_user.id}的服务{service_id}迁移")
                        logger.info(f"[紧急迁移] 卫星{agent.id}有紧急情况，跳过非紧急用户{service_user.id}的服务{service_id}迁移")
                        continue
                
                # 遍历最大目标卫星数
                for target_idx in range(max_target_sat_num):
                    # 要求2: 检查target_idx是否在agent.target_sat_list范围内
                    if target_idx >= len(agent.target_sat_list):
                        # 不满足要求2，该动作掩码保持为0
                        continue
                    
                    target_sat = agent.target_sat_list[target_idx]
                    
                    # 要求3: 检查目标卫星是否有足够的计算资源
                    if target_sat.comp_resource < service_size:
                        # 不满足要求3，跳过该目标卫星
                        continue
                    
                    # 要求4: 检查请求该服务的用户是否在目标卫星的可见用户列表中
                    if service_user not in target_sat.visible_user:
                        # 不满足要求4，跳过该目标卫星
                        continue
                    
                    # 额外要求5: 检查下一时刻用户对目标卫星的可见性
                    next_target_visibility = predict_next_step_visibility(service_user, target_sat, next_time)
                    if not next_target_visibility:
                        # 下一时刻对目标卫星也不可见，跳过该目标卫星
                        continue
                    
                    # 所有要求都满足，设置动作掩码为1
                    action_id = 1 + service_id * max_target_sat_num + target_idx
                    agent_available_actions[action_id] = 1.0
                    
                    
            # 检查是否有可用的迁移动作
            available_migration_actions = np.sum(agent_available_actions[1:])  # 排除动作0
            
            # 根据紧急情况设置动作0（不迁移）的可用性
            if has_emergency:
                if available_migration_actions > 0:
                    # 有紧急情况且有可用的迁移动作，强制选择迁移动作（动作0不可用）
                    print(f"[紧急迁移] 卫星{agent.id}需要紧急迁移且有{available_migration_actions}个可用迁移动作，强制选择迁移")
                    agent_available_actions[0] = 0.0  # 不迁移不可用
                else:
                    # 有紧急情况但没有可用的迁移动作，允许不迁移作为兜底
                    print(f"[警告] 卫星{agent.id}需要紧急迁移但没有可用的迁移动作！允许不迁移作为兜底")
                    agent_available_actions[0] = 1.0  # 允许不迁移作为兜底
            else:
                # 没有紧急情况，动作0（不迁移）是可用的
                agent_available_actions[0] = 1.0
            
            available_actions.append(agent_available_actions)
        return np.array(available_actions, dtype=np.float32)

    # step  this is  env.step()
    def step(self, action_n):
        """
        环境步进函数
        1. 为每个智能体设置动作
        2. 调用 world.step() 来推进物理仿真
        3. 获取新的观测、奖励、完成状态和信息
        Args:
            action_n: 策略网络输出的动作，action_n[i] 是第 i 个智能体的动作
        """
        # TODO：step步进是否需要

        # 定义obs reward
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.satellites
        
        # ====== DEBUG: 打印动作信息 ======
        print(f"[DEBUG] 环境step开始，接收到动作: {action_n}")
        print(f"[DEBUG] 动作类型: {type(action_n)}, 形状: {np.shape(action_n) if hasattr(action_n, 'shape') else 'N/A'}")
        logger.info(f"[DEBUG] 环境step开始，接收到动作: {action_n}")
        logger.debug(f"[DEBUG] 动作类型: {type(action_n)}, 形状: {np.shape(action_n) if hasattr(action_n, 'shape') else 'N/A'}")
        
        # 为每个智能体设置动作空间
        # action_n 是策略网络输出的动作，action_n[i] 是第 i 个智能体的动作
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)

        # 步进环境状态
        self.world.step()  # core.step()
        # 记录每个智能体的观测
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            # 计算奖励，避免重复调用
            agent_reward = self._get_reward(agent)
            reward_n.append([agent_reward])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': agent_reward}  # 使用已计算的奖励值
            print(f"[DEBUG] 卫星{agent.id}的奖励: {agent_reward}")
            logger.debug(f"[DEBUG] 卫星{agent.id}的奖励: {agent_reward}")
            env_info = self._get_info(agent) #似乎没啥用
          
            info_n.append(info)

        # 计算总的奖励，如果是shared-reward，则所有智能体共享奖励
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n
            print(f"[DEBUG] 共享奖励设置: {reward_n}")
            logger.info(f"[DEBUG] 共享奖励设置: {reward_n}")

        # 生成动作掩码
        available_actions = self.get_available_actions_with_emergency()

        # 生成共享观测（将所有智能体的观测连接起来）
        share_obs_n = []
        for i in range(self.n):
            # 对于每个智能体，共享观测是所有智能体观测的连接
            share_obs = np.concatenate(obs_n)
            share_obs_n.append(share_obs)

        # if self.post_step_callback is not None:
        #     self.post_step_callback(self.world)

        return obs_n, share_obs_n, reward_n, done_n, info_n, available_actions

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self):
        # 重置
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.satellites

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        # 生成初始动作掩码
        available_actions = self.get_available_actions_with_emergency()

        # 生成共享观测（将所有智能体的观测连接起来）
        share_obs_n = []
        for i in range(self.n):
            # 对于每个智能体，共享观测是所有智能体观测的连接
            share_obs = np.concatenate(obs_n)
            share_obs_n.append(share_obs)

        return obs_n, share_obs_n, available_actions

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        """获取单个智能体的观测"""
        # 对observation_callback的回调
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        """判断单个智能体是否结束"""
        if self.done_callback is None:
            # 默认的结束条件是达到最大步长
            return self.world.world_step >= self.world.world_length
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        """获取单个智能体的奖励"""
        if self.reward_callback is None:
            return 0.0
        
        reward = self.reward_callback(agent, self.world)
        logger.info(f"[DEBUG] 卫星{agent.id}: reward_callback返回奖励={reward}")
        return reward

    # set env action for a particular agent
    def _set_action(self, action_id, agent):
        '''
        将来自策略网络的整数动作，设置给对应的卫星智能体。
        '''
        # 如果action_id是one-hot编码的数组，转换为整数
        if isinstance(action_id, (list, np.ndarray)) and len(action_id) > 1:
            action_id = np.argmax(action_id)
        
        agent.action = SatelliteAction(action_id)

    def render(self, mode='html', save_dir=None):
        """
        调用 core.py 中强大的绘图函数来可视化环境。
        
        Args:
            mode: 渲染模式
                - 'html': 生成交互式HTML文件（默认）
                - 'png': 生成静态PNG图片
                - 'human': 实时显示（如果支持）
                - 'rgb_array': 返回RGB数组（用于录制视频）
            save_dir: 保存目录，如果为None则使用默认目录
        """
        if save_dir is None:
            if mode == 'html':
                save_dir = "satellite_steps_html"
            elif mode == 'png':
                save_dir = "satellite_steps_png"
            else:
                save_dir = "satellite_steps"
        
        if mode == 'html':
            # 调用交互式绘图函数
            self.world.plot_step_positions_interactive(self.world.world_step, save_dir)
            return None
        elif mode == 'png':
            # 调用静态绘图函数
            self.world.plot_step_positions(self.world.world_step, save_dir)
            return None
        elif mode == 'human':
            # 尝试实时显示（如果支持）
            try:
                self.world.plot_step_positions(self.world.world_step, save_dir)
                return None
            except Exception as e:
                logger.warning(f"实时显示不支持: {e}")
                return None
        elif mode == 'rgb_array':
            # 返回RGB数组（用于录制视频）
            # 这里需要实现从matplotlib图形转换为RGB数组
            try:
                import matplotlib.pyplot as plt
                import io
                from PIL import Image
                
                # 创建图形但不显示
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # 绘制地球
                r = 6371
                u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
                x = r * np.cos(u) * np.sin(v)
                y = r * np.sin(u) * np.sin(v)
                z = r * np.cos(v)
                ax.plot_surface(x, y, z, color='deepskyblue', alpha=0.3)
                
                # 绘制卫星
                sat_x, sat_y, sat_z = [], [], []
                for sat in self.world.satellites:
                    pos = sat._satellite_pos(self.world.current_time)
                    sat_x.append(pos[0])
                    sat_y.append(pos[1])
                    sat_z.append(pos[2])
                    ax.text(pos[0], pos[1], pos[2], f"S{sat.id}", fontsize=8, color='red')
                ax.scatter(sat_x, sat_y, sat_z, c='red', marker='o', label='Satellites')
                
                # 绘制用户
                user_x, user_y, user_z = [], [], []
                for user in self.world.user_clusters:
                    lon, lat = user.lon, user.lat
                    pos = [
                        r * np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                        r * np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                        r * np.sin(np.radians(lat))
                    ]
                    user_x.append(pos[0])
                    user_y.append(pos[1])
                    user_z.append(pos[2])
                    ax.text(pos[0], pos[1], pos[2], f"U{user.id}", fontsize=8, color='green')
                ax.scatter(user_x, user_y, user_z, c='green', marker='^', label='Users')
                
                ax.set_title(f"Step {self.world.world_step} 卫星与用户分布")
                ax.set_xlabel("X (km)")
                ax.set_ylabel("Y (km)")
                ax.set_zlabel("Z (km)")
                ax.legend()
                
                # 转换为RGB数组
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)
                rgb_array = np.array(img)
                plt.close(fig)
                
                return rgb_array
                
            except ImportError:
                logger.error("需要安装PIL库来支持rgb_array模式")
                return None
        else:
            logger.warning(f"不支持的渲染模式: {mode}")
            return None
