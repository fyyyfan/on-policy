from .environment import MultiAgentEnv
from .scenarios import load

'''
    对env的封装
'''
def SATEnv(args):
    '''
    

    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # 加载卫星场景
    scenario = load("satellite_scenario.py").Scenario()
    # create world 创建世界，scenario中的具体场景
    world = scenario.make_world(args)
    # 创建多智能体环境，environment.py中的具体对多智能体的处理
    # 传入scenario中定义的回调函数，奖励回调函数，观测回调函数，信息回调函数
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward_agent, scenario.observation_agent, scenario.info)

    return env