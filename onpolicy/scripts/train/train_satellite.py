#!/usr/bin/env python
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
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

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # 设置环境名称
    all_args.env_name = "Satellite"
    
    # 其他配置...
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy=True")
        all_args.use_recurrent_policy = True
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy=False")
        all_args.use_recurrent_policy = False
    else:
        raise NotImplementedError
    
    # cuda
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

    # run dir 结果存储路径
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb 可视化wandb配置
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         # project=all_args.env_name,
                         project=all_args.wandb_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                        # reinit=True
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

    # seed 设置随机数种子
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    ############
    #关键代码开始~ 创建环境
    # 创建环境
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    # 获取智能体数量
    num_agents = all_args.num_sats
    
    # 传给runner的 配置参数
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.satellite_runner import SatelliteRunner as Runner
    else:
        from onpolicy.runner.separated.satellite_runner import SatelliteRunner as Runner

    # 执行训练，不同场景下的Runner类不同
    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    # 直接写参数列表，和sh脚本里一模一样
    # debug_args = [
    #     "--env_name", "Satellite",
    #     "--algorithm_name", "rmappo",
    #     "--experiment_name", "check",
    #     "--scenario_name", "satellite_scenario",
    #     "--num_sats", "6",
    #     "--num_users", "5",
    #     "--seed", "1",
    #     "--n_training_threads", "1",
    #     "--n_rollout_threads", "1",
    #     "--num_mini_batch", "1",
    #     "--episode_length", "8",
    #     "--num_env_steps", "20000000",
    #     "--ppo_epoch", "10",
    #     "--use_ReLU",
    #     "--gain", "0.01",
    #     "--lr", "7e-4",
    #     "--critic_lr", "7e-4",
    #     "--wandb_name", "Satellite-personalTest",
    #     "--user_name", "fyyyfan06-uestc",
    #     "--h", "7000.0",
    #     "--angle", "51.664",
    #     "--P_num", "1",
    #     "--dt", "60.0",
    #     "--sat_comp_resource", "[100.0,100.0,100.0,100.0,100.0,100.0]",
    #     "--sat_tran_power", "[10.0,10.0,10.0,10.0,10.0,10.0]",
    #     "--sat_tran_gain", "[1.0,1.0,1.0,1.0,1.0,1.0]",
    #     "--sat_rec_gain", "[1.0,1.0,1.0,1.0,1.0,1.0]",
    #     "--user_lon", "[100.0,120.0,140.0,160.0,180.0]",
    #     "--user_lat", "[40.0,45.0,50.0,35.0,30.0]",
    #     "--start_time", "[2024,1,3,8,0,0]"
    # ]
    # main(debug_args)
    # 如果需要运行sh脚本，则取消注释
    main(sys.argv[1:])
