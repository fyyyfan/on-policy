#!/bin/sh

# 设置Python路径，确保能找到onpolicy模块
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# 评估配置
env="Satellite"
scenario="satellite_scenario" 
algo="rmappo"  # 可选: "rmappo", "mappo", "ippo"
exp="satellite_evaluation"
seed=1

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

# 评估配置
n_training_threads=1
n_rollout_threads=8      # 训练环境线程数
n_eval_rollout_threads=4 # 评估环境线程数（并行评估）
episode_length=25         # Episode长度
use_eval=True            # 启用评估模式
use_wandb=False          # 是否使用wandb记录

# 模型路径 - 必须指定训练好的模型目录
model_dir="results/Satellite/satellite_scenario/rmappo/satellite_training/run1/models"  # 请根据实际路径修改

echo "=== Satellite环境评估配置 ==="
echo "env: ${env}, scenario: ${scenario}, algo: ${algo}, exp: ${exp}"
echo "num_sats: ${num_sats}, num_users: ${num_users}"
echo "episode_length: ${episode_length}, n_eval_rollout_threads: ${n_eval_rollout_threads}"
echo "model_dir: ${model_dir}"

# 检查模型目录是否存在
if [ ! -d "$model_dir" ]; then
    echo "错误: 模型目录不存在: $model_dir"
    echo "请先训练模型或修改model_dir路径"
    exit 1
fi

echo "开始执行Satellite环境评估..."

# 运行评估脚本
python eval_satellite.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --seed ${seed} \
    --cuda False \
    --n_training_threads ${n_training_threads} \
    --n_rollout_threads ${n_rollout_threads} \
    --n_eval_rollout_threads ${n_eval_rollout_threads} \
    --episode_length ${episode_length} \
    --use_eval ${use_eval} \
    --use_wandb ${use_wandb} \
    --wandb_name "Satellite-Evaluation" \
    --user_name "evaluation_user" \
    --model_dir ${model_dir} \
    --share_policy True \
    --use_centralized_V True \
    --hidden_size 64 \
    --layer_N 1 \
    --use_ReLU \
    --use_valuenorm \
    --use_feature_normalization \
    --use_orthogonal \
    --gain 0.01 \
    --use_recurrent_policy False \
    --recurrent_N 1 \
    --data_chunk_length 10 \
    --h ${h} \
    --angle ${angle} \
    --P_num ${P_num} \
    --dt ${dt} \
    --num_sats ${num_sats} \
    --num_users ${num_users} \
    --sat_comp_resource ${sat_comp_resource} \
    --sat_tran_power ${sat_tran_power} \
    --sat_tran_gain ${sat_tran_gain} \
    --sat_rec_gain ${sat_rec_gain} \
    --user_lon ${user_lon} \
    --user_lat ${user_lat} \
    --start_time ${start_time}

echo "Satellite环境评估完成！"
echo "结果保存在: results/${env}/${scenario}/${algo}/${exp}/" 