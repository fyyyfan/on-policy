#!/bin/sh
# 设置Python路径，确保能找到onpolicy模块

##场景1：user-lon [100.0,120.0,140.0,160.0,180.0] user_lat "[40.0,45.0,50.0,35.0,30.0]" v1:--episode_length 50 --dt 10.0
##场景2："[100.0, 110.0, 120.0, 125.0, 130.0]"[40.0, 50.0, 47.0, 42.0, 45.0] 
#场景3： --user_lon "[100.0, 110.0, 120.0, 130.0, 105.0, 115.0, 125.0, 135.0]" \--user_lat "[40.0, 50.0, 47.0, 45.0, 42.0, 35.0, 30.0, 25.0]" \
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

env="Satellite"
scenario="satellite_scenario" 
num_sats=4
num_users=4
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_satellite.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_sats ${num_sats} --num_users ${num_users} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 8 --episode_length 20 --num_env_steps 600000  --entropy_coef 0.015 --gamma 0.995\
    --log_interval 2 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 5e-5 --critic_lr 5e-5 --wandb_name "Satellite-personalTest" --user_name "fyyyfan06-uestc" \
    --h 7000.0 --angle 51.664 --P_num 1 --dt 20.0 \
    --sat_comp_resource "[100.0,100.0,100.0,100.0]" \
    --sat_tran_power "[10.0,10.0,10.0,10.0]" \
    --sat_tran_gain "[36.0,36.0,36.0,36.0]" \
    --sat_rec_gain "[41.0,41.0,41.0,41.0]" \
    --user_lon "[100.0, 110.0, 120.0, 130.0]" \
    --user_lat "[40.0, 50.0, 47.0, 45.0]" \
    --start_time "[2024,1,3,8,0,0]"
done 