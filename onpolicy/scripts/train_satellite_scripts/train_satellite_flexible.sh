#!/bin/sh

# 设置Python路径，确保能找到onpolicy模块
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# 环境配置
env="Satellite"
scenario="satellite_scenario" 
algo="rmappo" #"mappo" "ippo"
exp="satellite_test"
seed_max=1

# 卫星配置
num_sats=6
h=7000.0  # 轨道半径(km)
angle=0.0  # 轨道倾角(度)
P_num=1    # 轨道面数
dt=60.0    # 时间步长(秒)

# 用户配置
num_users=5

# 卫星资源配置 (需要与num_sats匹配)
sat_comp_resource="\"[100.0,100.0,100.0,100.0,100.0,100.0]\""
sat_tran_power="\"[10.0,10.0,10.0,10.0,10.0,10.0]\""
sat_tran_gain="\"[1.0,1.0,1.0,1.0,1.0,1.0]\""
sat_rec_gain="\"[1.0,1.0,1.0,1.0,1.0,1.0]\""

# 用户位置配置 (需要与num_users匹配)
user_lon="\"[100.0,120.0,140.0,160.0,180.0]\""
user_lat="\"[40.0,45.0,50.0,35.0,30.0]\""

# 时间配置
start_time="\"[2024,1,3,8,0,0]\""

# 训练配置
n_training_threads=1
n_rollout_threads=128
num_mini_batch=1
episode_length=25
num_env_steps=20000000
ppo_epoch=10
lr=7e-4
critic_lr=7e-4

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "num_sats: ${num_sats}, num_users: ${num_users}"
echo "h: ${h}, angle: ${angle}, P_num: ${P_num}, dt: ${dt}"

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
    --wandb_name "Satellite-personalTest" \
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