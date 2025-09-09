#!/bin/sh

# 设置Python路径，确保能找到onpolicy模块
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# 调试配置 - 快速测试
env="Satellite"
scenario="satellite_scenario" 
algo="mappo"  # 使用mappo进行调试，避免RNN的复杂性
exp="satellite_debug"
seed_max=1

# 简化配置 - 少量卫星和用户
num_sats=3
num_users=2
h=7000.0
angle=0.0
P_num=1
dt=60.0

# 简化的资源配置
sat_comp_resource="\"[100.0,100.0,100.0]\""
sat_tran_power="\"[10.0,10.0,10.0]\""
sat_tran_gain="\"[1.0,1.0,1.0]\""
sat_rec_gain="\"[1.0,1.0,1.0]\""

# 简化的用户位置
user_lon="\"[100.0,120.0]\""
user_lat="\"[40.0,45.0]\""

# 时间配置
start_time="\"[2024,1,3,8,0,0]\""

# 调试训练配置 - 快速收敛
n_training_threads=1
n_rollout_threads=4  # 减少并行环境数量
num_mini_batch=1
episode_length=10    # 缩短episode长度
num_env_steps=10000  # 减少总训练步数
ppo_epoch=5          # 减少PPO epoch
lr=7e-4
critic_lr=7e-4

echo "=== 卫星环境调试模式 ==="
echo "env: ${env}, scenario: ${scenario}, algo: ${algo}, exp: ${exp}"
echo "num_sats: ${num_sats}, num_users: ${num_users}"
echo "episode_length: ${episode_length}, num_env_steps: ${num_env_steps}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../train/train_satellite.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_sats ${num_sats} \
    --num_users ${num_users} \
    --seed ${seed} \
    --n_training_threads ${n_training_threads} \
    --n_rollout_threads ${n_rollout_threads} \
    --num_mini_batch ${num_mini_batch} \
    --episode_length ${episode_length} \
    --num_env_steps ${num_env_steps} \
    --ppo_epoch ${ppo_epoch} \
    --use_ReLU \
    --gain 0.01 \
    --lr ${lr} \
    --critic_lr ${critic_lr} \
    --wandb_name "Satellite-Debug" \
    --user_name "fyyyfan06-uestc" \
    --h ${h} \
    --angle ${angle} \
    --P_num ${P_num} \
    --dt ${dt} \
    --sat_comp_resource ${sat_comp_resource} \
    --sat_tran_power ${sat_tran_power} \
    --sat_tran_gain ${sat_tran_gain} \
    --sat_rec_gain ${sat_rec_gain} \
    --user_lon ${user_lon} \
    --user_lat ${user_lat} \
    --start_time ${start_time}
done 