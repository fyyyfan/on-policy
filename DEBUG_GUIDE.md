# 卫星环境调试指南

## 概述

本文档提供了调试卫星环境训练脚本的方法和工具。

## 调试脚本

### 1. 环境测试脚本 (`debug_env_only.py`)

这个脚本专门用于测试环境创建和基本功能，不涉及完整的训练流程。

**使用方法：**
```bash
python debug_env_only.py
```

**功能：**
- 测试环境创建
- 验证观测空间和动作空间
- 测试环境重置和步进
- 检查智能体数量设置

### 2. 完整训练调试脚本 (`debug_train_satellite.py`)

这个脚本模拟完整的训练流程，包含所有必需的参数。

**使用方法：**
```bash
python debug_train_satellite.py
```

**功能：**
- 完整的环境创建
- Runner创建和训练
- 错误处理和资源清理

## 关键参数说明

### 卫星环境特有参数

```python
# 卫星配置
--num_sats 4                    # 卫星数量
--h 7000.0                     # 轨道高度(km)
--angle 0.0                    # 轨道倾角(度)
--P_num 1                      # 轨道面数

# 卫星资源参数
--sat_comp_resource 100.0 100.0 100.0 100.0  # 计算资源
--sat_tran_power 10.0 10.0 10.0 10.0         # 传输功率
--sat_tran_gain 1.0 1.0 1.0 1.0              # 传输增益
--sat_rec_gain 1.0 1.0 1.0 1.0               # 接收增益

# 用户配置
--num_users 3                                  # 用户数量
--user_lon 100.0 120.0 140.0                  # 用户经度
--user_lat 40.0 45.0 50.0                     # 用户纬度

# 时间配置
--start_time 2024 1 3 8 0 0                   # 开始时间
--dt 60.0                                     # 时间步长(秒)
--episode_length 50                           # Episode长度
```

### 调试优化参数

```python
# 调试时建议的设置
--cuda False                    # 使用CPU调试
--n_rollout_threads 1          # 单线程
--num_env_steps 1000           # 减少训练步数
--use_wandb False              # 关闭wandb
--use_eval False               # 关闭评估
--use_render False             # 关闭渲染
--use_recurrent_policy False   # 关闭循环网络
```

## 常见问题排查

### 1. 智能体数量为0

**问题：** `IndexError: list index out of range`

**原因：** `world.num_agents` 没有正确设置

**解决方案：** 确保在 `make_world` 函数中添加：
```python
world.num_agents = len(world.satellites)
```

### 2. 观测空间为空

**问题：** `observation_space` 和 `share_observation_space` 为空列表

**原因：** 智能体数量为0，导致初始化循环不执行

**解决方案：** 检查 `self.agents` 和 `self.n` 是否正确设置

### 3. 环境包装器不兼容

**问题：** `SubprocVecEnv` 期望的返回格式不匹配

**解决方案：** 使用支持动作掩码的包装器：
```python
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
```

## 调试步骤

1. **首先运行环境测试：**
   ```bash
   python debug_env_only.py
   ```

2. **检查输出：**
   - 确认智能体数量正确
   - 确认观测空间和动作空间有值
   - 确认环境重置和步进正常

3. **如果环境测试通过，运行完整训练：**
   ```bash
   python debug_train_satellite.py
   ```

4. **逐步调试：**
   - 在关键位置添加打印语句
   - 使用 try-except 捕获具体错误
   - 检查参数传递是否正确

## 添加调试信息

在代码中添加调试信息：

```python
# 在环境初始化时
print(f"智能体数量: {self.n}")
print(f"观测空间: {len(self.observation_space)}")
print(f"共享观测空间: {len(self.share_observation_space)}")

# 在make_world中
print(f"卫星列表长度: {len(world.satellites)}")
print(f"设置的智能体数量: {world.num_agents}")

# 在reset中
print(f"重置后的智能体数量: {len(self.agents)}")
```

## 参数验证

确保以下参数列表长度匹配：

```python
# 卫星相关参数长度必须等于 num_sats
len(sat_comp_resource) == num_sats
len(sat_tran_power) == num_sats
len(sat_tran_gain) == num_sats
len(sat_rec_gain) == num_sats

# 用户相关参数长度必须等于 num_users
len(user_lon) == num_users
len(user_lat) == num_users
```

## 注意事项

1. **调试时使用小规模配置：** 减少卫星数量、用户数量、episode长度
2. **关闭不必要的功能：** wandb、评估、渲染等
3. **使用CPU调试：** 避免GPU相关问题
4. **单线程运行：** 避免多进程相关问题
5. **逐步增加复杂度：** 先确保基本功能正常，再增加参数 