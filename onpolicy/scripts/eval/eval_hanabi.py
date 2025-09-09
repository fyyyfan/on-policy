#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config

from onpolicy.envs.hanabi.Hanabi_Env import HanabiEnv
from onpolicy.envs.env_wrappers import ChooseSubprocVecEnv, ChooseDummyVecEnv


def make_train_env(all_args):
    """
    创建训练环境的工厂函数
    支持多线程并行环境创建
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                # Hanabi游戏只支持2-5个智能体
                assert all_args.num_agents > 1 and all_args.num_agents < 6, (
                    "num_agents can be only between 2-5.")
                # 为每个线程创建独立的Hanabi环境，使用不同的随机种子
                env = HanabiEnv(all_args, (all_args.seed + rank * 1000))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    # 根据线程数选择环境类型：单线程用DummyVecEnv，多线程用SubprocVecEnv
    if all_args.n_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """
    创建评估环境的工厂函数
    使用不同的随机种子确保评估的独立性
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                # 评估环境使用更大的种子偏移，确保与训练环境完全独立
                assert all_args.num_agents > 1 and all_args.num_agents < 6, (
                    "num_agents can be only between 2-5.")
                env = HanabiEnv(
                    all_args, (all_args.seed * 50000 + rank * 10000))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    # 根据评估线程数选择环境类型
    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """
    解析Hanabi特定的命令行参数
    """
    # 指定Hanabi环境的具体变体（如Very-Small, Small等）
    parser.add_argument('--hanabi_name', type=str,
                        default='Hanabi-Very-Small', help="Which env to run on")
    # 设置游戏中的智能体数量（玩家数量）
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # 获取基础配置
    parser = get_config()
    all_args = parse_args(args, parser)

    # 根据选择的算法自动配置相关参数
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        # R-MAPPO启用循环策略，适合处理部分可观察性
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        # MAPPO禁用循环策略，使用标准策略网络
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        # IPPO禁用中心化价值函数，每个智能体独立学习
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # 验证评估模式已启用
    assert all_args.use_eval, ("u need to set use_eval be True")
    # 验证模型目录已设置（用于加载训练好的模型）
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    

    # GPU/CPU设备配置
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            # 启用确定性计算，确保结果可重复
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 创建结果存储目录
    # 目录结构：results/Hanabi/hanabi_name/algorithm_name/experiment_name/
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.hanabi_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Wandb实验跟踪配置
    if all_args.use_wandb:
        # 初始化wandb实验，记录配置和结果
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.hanabi_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        # 本地实验管理：自动创建run1, run2等子目录
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

    # 设置进程标题，便于系统监控
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # 设置随机种子，确保实验可重复性
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 初始化环境
    envs = make_train_env(all_args)  # 训练环境（用于模型推理）
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None  # 评估环境
    num_agents = all_args.num_agents

    # 构建配置字典，传递给Runner
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # 根据策略共享模式选择相应的Runner
    if all_args.share_policy:
        # 共享策略：所有智能体使用相同的策略网络
        from onpolicy.runner.shared.hanabi_runner_forward import HanabiRunner as Runner
    else:
        # 分离策略：每个智能体使用独立的策略网络
        from onpolicy.runner.separated.hanabi_runner_forward import HanabiRunner as Runner

    # 创建Runner实例并执行评估
    runner = Runner(config)
    # 执行100k步的评估，收集性能指标
    runner.eval_100k()
    
    # 清理资源
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # 完成实验记录
    if all_args.use_wandb:
        run.finish()
    else:
        # 导出本地日志数据
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

