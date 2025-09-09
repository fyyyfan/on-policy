# Satellite环境评估脚本使用说明

## 概述

本评估脚本专门用于评估Satellite环境中强化学习模型的性能，主要关注两个关键指标：
1. **所有用户的平均服务延迟**：包括计算延迟和通信延迟
2. **所有用户的平均迁移延迟**：服务实例在不同卫星间迁移的延迟

## 主要功能

### 1. 延迟数据收集
- 自动收集每个episode中所有用户的延迟数据
- 支持多线程并行环境评估
- 每5个时间步收集一次数据，避免数据过多

### 2. 评估指标计算
- **服务延迟** = 计算延迟 + 通信延迟
- **迁移延迟** = 实例传输延迟 + 传播延迟 + 服务停止/启动时间
- 提供延迟的统计信息：平均值、标准差、最小值、最大值

### 3. 结果输出
- 控制台实时显示每个episode的延迟统计
- 保存详细的延迟数据到CSV文件
- 支持Wandb实验跟踪（可选）

## 使用方法

### 1. 基本运行命令

```bash
# 使用shell脚本运行（推荐）
bash eval_satellite.sh

# 或直接使用Python脚本
python eval_satellite.py [参数列表]
```

### 2. 必需参数

```bash
--env_name Satellite                    # 环境名称
--algorithm_name rmappo                # 算法名称（rmappo/mappo/ippo）
--experiment_name satellite_evaluation # 实验名称
--model_dir /path/to/trained/model    # 训练好的模型路径（必需）
--use_eval True                       # 启用评估模式
```

### 3. 卫星环境特有参数

```bash
# 卫星配置
--num_sats 6                          # 卫星数量
--h 7000.0                           # 轨道高度(km)
--angle 0.0                          # 轨道倾角(度)
--P_num 1                            # 轨道面数
--dt 60.0                            # 时间步长(秒)

# 用户配置
--num_users 5                        # 用户数量
--user_lon "[100.0,120.0,140.0,160.0,180.0]"  # 用户经度列表
--user_lat "[40.0,45.0,50.0,35.0,30.0]"       # 用户纬度列表

# 卫星资源配置
--sat_comp_resource "[100.0,100.0,100.0,100.0,100.0,100.0]"  # 计算资源
--sat_tran_power "[10.0,10.0,10.0,10.0,10.0,10.0]"           # 传输功率
--sat_tran_gain "[1.0,1.0,1.0,1.0,1.0,1.0]"                 # 传输增益
--sat_rec_gain "[1.0,1.0,1.0,1.0,1.0,1.0]"                  # 接收增益
```

### 4. 评估配置参数

```bash
--n_eval_rollout_threads 4           # 评估环境并行数量
--episode_length 25                  # 每个episode的时间步数
--use_wandb False                    # 是否使用wandb记录
--cuda False                         # 是否使用GPU
```

## 输出结果

### 1. 控制台输出

```
=== Satellite环境评估配置 ===
env: Satellite, scenario: satellite_scenario, algo: rmappo, exp: satellite_evaluation
num_sats: 6, num_users: 5
episode_length: 25, n_eval_rollout_threads: 4
model_dir: results/Satellite/satellite_scenario/rmappo/satellite_training/run1/models

开始执行Satellite环境评估...
评估Episode 1/10
Episode 1 平均服务延迟: 45.234 ms
Episode 1 平均迁移延迟: 12.567 ms
...

=== 总体评估结果 ===
所有用户平均服务延迟: 47.891 ms
服务延迟标准差: 8.234 ms
服务延迟范围: [32.123, 65.789] ms
所有用户平均迁移延迟: 13.456 ms
迁移延迟标准差: 2.345 ms
迁移延迟范围: [8.901, 18.234] ms
```

### 2. 文件输出

评估结果保存在以下目录结构：
```
results/
└── Satellite/
    └── satellite_scenario/
        └── rmappo/
            └── satellite_evaluation/
                └── run1/
                    ├── evaluation_delay_data.csv    # 详细延迟数据
                    └── wandb/                       # Wandb日志（如果启用）
```

### 3. CSV数据格式

`evaluation_delay_data.csv` 包含以下列：
- `episode`: Episode编号
- `step`: 时间步编号
- `avg_service_delay_ms`: 平均服务延迟（毫秒）
- `avg_migration_delay_ms`: 平均迁移延迟（毫秒）
- `service_count`: 有效服务数量
- `migration_count`: 迁移事件数量

## 注意事项

### 1. 模型路径
- 必须指定 `--model_dir` 参数，指向训练好的模型目录
- 模型目录应包含 `actor.pt` 和 `critic.pt` 文件

### 2. 环境配置
- 确保卫星数量、用户数量与资源配置列表长度匹配
- 用户位置坐标应在合理范围内

### 3. 性能优化
- 使用 `n_eval_rollout_threads` 控制并行评估数量
- 调整 `episode_length` 平衡评估精度和速度
- 每5步收集一次数据，可根据需要调整

### 4. 错误处理
- 脚本会自动检查模型目录是否存在
- 延迟数据收集失败时会记录错误并继续执行
- 支持CPU和GPU运行模式

## 示例配置

### 快速评估配置
```bash
--n_eval_rollout_threads 2    # 减少并行数量
--episode_length 10           # 缩短episode长度
--use_wandb False             # 关闭wandb
--cuda False                  # 使用CPU
```

### 高精度评估配置
```bash
--n_eval_rollout_threads 8    # 增加并行数量
--episode_length 50           # 延长episode长度
--use_wandb True              # 启用wandb记录
--cuda True                   # 使用GPU
```

## 故障排除

### 1. 常见错误
- **模型目录不存在**: 检查 `--model_dir` 路径是否正确
- **环境创建失败**: 检查卫星和用户配置参数
- **内存不足**: 减少 `n_eval_rollout_threads` 数量

### 2. 调试建议
- 先使用单线程环境测试：`--n_eval_rollout_threads 1`
- 检查环境参数是否与训练时一致
- 查看控制台输出的详细错误信息

## 扩展功能

### 1. 自定义评估指标
可以在 `collect_delay_data` 函数中添加其他指标：
- 服务成功率
- 资源利用率
- 链路质量指标

### 2. 多模型对比
可以修改脚本支持批量评估多个模型：
- 遍历模型目录
- 对比不同模型的性能
- 生成对比报告

### 3. 可视化支持
可以集成matplotlib等库：
- 绘制延迟变化曲线
- 生成性能对比图表
- 导出评估报告 