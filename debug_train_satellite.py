#!/usr/bin/env python
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "."))
sys.path.insert(0, project_root)

import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.satellite.SAT_env import SATEnv
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

def make_train_env(all_args):
    """
    创建训练环境
    Args:
        需要包含scenario.make_world(args)的所有参数
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Satellite":
                env = SATEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    """创建评估环境"""
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Satellite":
                env = SATEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    '''
    解析命令行参数,给出默认值
    '''
    parser.add_argument('--scenario_name', type=str,
                        default='satellite_scenario', help="Which scenario to run on")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main():
    """调试训练主函数"""
    
    # 创建参数解析器
    parser = get_config()
    
    # 设置调试参数（模拟命令行参数）
    debug_args = [
        # 基础参数
        "--algorithm_name", "mappo",
        "--experiment_name", "debug_test",
        "--seed", "1",
        "--cuda", "False",  # 调试时使用CPU
        "--n_training_threads", "1",
        "--n_rollout_threads", "1",  # 调试时使用单线程
        "--n_eval_rollout_threads", "1",
        "--num_env_steps", "1000",  # 调试时减少步数
        "--use_wandb", "False",  # 调试时关闭wandb
        "--wandb_name", "debug_test",
        "--user_name", "debug_user",
        
        # 环境参数
        "--env_name", "Satellite",
        "--scenario_name", "satellite_scenario",
        "--episode_length", "50",  # 调试时减少episode长度
        
        # 卫星环境特有参数
        "--start_time", "[2024,1,3,8,0,0]",
        "--dt", "60.0",
        "--num_users", "3",  # 调试时减少用户数
        "--user_lon", "[110.0, 120.0, 140.0]",
        "--user_lat", "[40.0, 45.0, 50.0]",
        "--num_sats", "4",  # 调试时减少卫星数
        "--h", "7000.0",
        "--angle", "0.0",
        "--P_num", "1",
        "--sat_comp_resource", "[100.0,100.0,100.0,100.0]",
        "--sat_tran_power", "[10.0,10.0,10.0,10.0]",
        "--sat_tran_gain", "[1.0,1.0,1.0,1.0]",
        "--sat_rec_gain", "[1.0,1.0,1.0,1.0]",
        
        # 网络参数
        "--share_policy", "True",
        "--use_centralized_V", "True",
        "--hidden_size", "64",
        "--layer_N", "1",
        "--use_ReLU", "True",
        "--use_valuenorm", "True",
        "--use_feature_normalization", "True",
        "--use_orthogonal", "True",
        "--gain", "0.01",
        
        # 循环网络参数
        "--use_recurrent_policy", "False",  # 调试时关闭循环网络
        "--recurrent_N", "1",
        "--data_chunk_length", "10",
        
        # 优化器参数
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--opti_eps", "1e-5",
        "--weight_decay", "0",
        
        # PPO参数
        "--ppo_epoch", "15",
        "--use_clipped_value_loss", "True",
        "--clip_param", "0.2",
        "--num_mini_batch", "1",
        "--entropy_coef", "0.01",
        "--use_max_grad_norm", "True",
        "--max_grad_norm", "10.0",
        "--use_gae", "True",
        "--gamma", "0.99",
        "--gae_lambda", "0.95",
        "--use_proper_time_limits", "False",
        "--use_huber_loss", "True",
        "--use_value_active_masks", "True",
        "--use_policy_active_masks", "True",
        "--huber_delta", "10.0",
        
        # 运行参数
        "--use_linear_lr_decay", "False",
        
        # 保存和日志参数
        "--save_interval", "1",
        "--log_interval", "5",
        
        # 评估参数
        "--use_eval", "False",  # 调试时关闭评估
        "--eval_interval", "25",
        "--eval_episodes", "32",
        
        # 渲染参数
        "--save_gifs", "False",
        "--use_render", "False",
        "--render_episodes", "5",
        "--ifi", "0.1",
        
        # 预训练参数
        "--model_dir", "None",
    ]
    
    all_args = parse_args(debug_args, parser)
    
    # 设置环境名称
    all_args.env_name = "Satellite"
    
    # 算法配置
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy=True")
        all_args.use_recurrent_policy = True
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy=False")
        all_args.use_recurrent_policy = False
    else:
        raise NotImplementedError
    
    # cuda配置
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 运行目录
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb配置（调试时关闭）
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.wandb_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         settings=wandb.Settings(start_method="thread")
                        )
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # 设置随机种子
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    print("开始创建环境...")
    
    # 创建环境
    try:
        envs = make_train_env(all_args)
        print("训练环境创建成功")
    except Exception as e:
        print(f"训练环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    # 获取智能体数量
    num_agents = all_args.num_sats
    print(f"智能体数量: {num_agents}")
    
    # 传给runner的配置参数
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    print("开始创建Runner...")
    
    # 创建Runner
    try:
        # 卫星环境只支持共享策略，统一从shared模块导入
        from onpolicy.runner.shared.satellite_runner import SatelliteRunner as Runner

        runner = Runner(config)
        print("Runner创建成功")
    except Exception as e:
        print(f"Runner创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("开始运行训练...")
    
    # 执行训练
    try:
        runner.run()
        print("训练完成")
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()

        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()

if __name__ == "__main__":
    main() 