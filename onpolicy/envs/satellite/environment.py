import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from .core import SatelliteWorld, SatelliteAction
# update bounds to center around agent
cam_range = 2

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
            agent.action.c = np.zeros(self.world.dim_c) #?
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # # 5. 渲染相关（可选）
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()


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
        # 为每个智能体设置动作空间
        # action_n 是策略网络输出的动作，action_n[i] 是第 i 个智能体的动作
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)

        # 步进环境状态
        self.world.step()  # core.step()
        # 记录每个智能体的观测
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
          
            info_n.append(info)

        # 计算总的奖励，如果是shared-reward，则所有智能体共享奖励
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        # if self.post_step_callback is not None:
        #     self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # 重置
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.satellites

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

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
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action_id, agent):
        '''
        将来自策略网络的整数动作，设置给对应的卫星智能体。
        '''
        # action 是一个整数ID
        agent.action = SatelliteAction(action_id)

    def render(self, mode='html'):
        """
        调用 core.py 中强大的绘图函数来可视化环境。
        """
        if mode == 'html':
            # 调用您在 core.py 中定义的交互式绘图函数
            self.world.plot_step_positions_interactive(self.world.world_step)
        elif mode == 'human':
            # 调用 matplotlib 绘图函数
            self.world.plot_step_positions(self.world.world_step)
