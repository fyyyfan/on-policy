# 卫星环境接口修复总结

## 问题分析

原始报错：`ValueError: too many values to unpack (expected 2)`

**根本原因**：卫星环境的接口与其他环境不匹配，导致 `satellite_runner.py` 中的返回值解包失败。

## 具体问题

### 1. 环境接口不匹配

**卫星环境的接口：**
- `reset()` 返回：`obs_n, share_obs_n, available_actions` (3个值)
- `step()` 返回：`obs_n, share_obs_n, reward_n, done_n, info_n, available_actions` (6个值)

**Runner期望的接口：**
- `reset()` 期望：`obs, available_actions` (2个值)
- `step()` 期望：`obs, rewards, dones, infos, available_actions` (5个值)

### 2. 修复内容

#### 2.1 `warmup` 方法修复
```python
# 修复前
obs, available_actions = self.envs.reset()

# 修复后
obs, share_obs, available_actions = self.envs.reset()
```

#### 2.2 `run` 方法修复
```python
# 修复前
obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions

# 修复后
obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions
```

#### 2.3 `insert` 方法修复
```python
# 修复前
obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data

# 修复后
obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data
```

#### 2.4 `eval` 方法修复
```python
# 修复前
eval_obs, eval_available_actions = self.eval_envs.reset()
eval_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions_env)

# 修复后
eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions_env)
```

#### 2.5 `render` 方法修复
```python
# 修复前
obs, available_actions = envs.reset()
obs, rewards, dones, infos, available_actions = envs.step(actions_env)

# 修复后
obs, share_obs, available_actions = envs.reset()
obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions_env)
```

### 3. 共享观测处理

修复了共享观测的处理逻辑：
```python
# 修复前（错误的重新计算）
if self.use_centralized_V:
    share_obs = obs.reshape(self.n_rollout_threads, -1)
    share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

# 修复后（直接使用环境返回的共享观测）
if self.use_centralized_V:
    share_obs = share_obs
```

## 修复文件

- `onpolicy/runner/shared/satellite_runner.py`

## 验证方法

1. 运行环境测试脚本：
   ```bash
   python debug_env_only.py
   ```

2. 运行完整训练脚本：
   ```bash
   python debug_train_satellite.py
   ```

## 注意事项

1. **接口一致性**：确保卫星环境的接口与其他环境保持一致
2. **共享观测**：卫星环境已经计算好共享观测，不需要重新计算
3. **动作掩码**：卫星环境支持动作掩码，需要正确处理
4. **返回值数量**：注意 `reset` 和 `step` 方法的返回值数量

## 后续建议

1. **统一接口标准**：建议为所有环境定义统一的接口标准
2. **接口文档**：为每个环境编写详细的接口文档
3. **测试覆盖**：增加更多的接口测试用例
4. **错误处理**：在接口不匹配时提供更清晰的错误信息 