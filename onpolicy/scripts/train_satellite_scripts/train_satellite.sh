#!/bin/sh
# 设置Python路径，确保能找到onpolicy模块
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

env="Satellite"
scenario="satellite_scenario" 
num_sats=6
num_users=5
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_satellite.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_sats ${num_sats} --num_users ${num_users} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 8 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "Satellite-personalTest" --user_name "fyyyfan06-uestc" \
    --h 7000.0 --angle 51.664 --P_num 1 --dt 60.0 \
    --sat_comp_resource "[100.0,100.0,100.0,100.0,100.0,100.0]" \
    --sat_tran_power "[10.0,10.0,10.0,10.0,10.0,10.0]" \
    --sat_tran_gain "[1.0,1.0,1.0,1.0,1.0,1.0]" \
    --sat_rec_gain "[1.0,1.0,1.0,1.0,1.0,1.0]" \
    --user_lon "[100.0,120.0,140.0,160.0,180.0]" \
    --user_lat "[40.0,45.0,50.0,35.0,30.0]" \
    --start_time "[2024,1,3,8,0,0]"
done 