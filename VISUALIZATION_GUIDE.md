# 卫星环境可视化功能使用指南

## 概述

本指南介绍如何在卫星环境中使用可视化功能，包括训练过程中的阶段性可视化和迁移策略的可视化展示。

## 功能特性

### 1. 多种渲染模式

- **HTML模式** (`mode='html'`): 生成交互式HTML文件，支持3D旋转、缩放等交互操作
- **PNG模式** (`mode='png'`): 生成静态PNG图片，适合报告和文档
- **RGB数组模式** (`mode='rgb_array'`): 返回RGB数组，用于录制视频或进一步处理
- **Human模式** (`mode='human'`): 实时显示（如果支持）

### 2. 训练过程可视化

- **阶段性可视化**: 在训练过程中定期保存环境状态的可视化
- **迁移统计**: 记录和分析迁移动作的分布
- **性能摘要**: 生成episode级别的性能摘要图表

## 使用方法

### 1. 基础可视化

```python
from onpolicy.envs.satellite.environment import MultiAgentEnv

# 创建环境
env = MultiAgentEnv(...)

# 生成HTML交互式可视化
env.render(mode='html', save_dir='my_visualization')

# 生成PNG静态图片
env.render(mode='png', save_dir='my_images')

# 获取RGB数组
rgb_array = env.render(mode='rgb_array')
```

### 2. 训练过程中的可视化

在训练脚本中添加可视化参数：

```bash
python train_satellite.py --enable_visualization
```

或者在代码中设置：

```python
# 在SatelliteRunner中
self.all_args.enable_visualization = True
```

### 3. 自定义可视化间隔

```python
# 在SatelliteRunner中修改
viz_save_interval = 50  # 每50步保存一次可视化
```

## 输出文件结构

### 训练可视化目录结构

```
training_visualization/
├── episode_0001/
│   ├── step_0050_env_0.html          # HTML交互式可视化
│   ├── step_0050_env_0.png           # PNG静态图片
│   ├── step_0050_env_0_status.json   # 状态信息
│   ├── step_0100_env_0.html
│   └── ...
├── episode_0002/
│   └── ...
└── ...
```

### 训练摘要目录结构

```
training_summary/
├── episode_0010/
│   └── episode_0010_summary.png      # 性能摘要图表
├── episode_0020/
│   └── episode_0020_summary.png
└── ...
```

## 可视化内容

### 1. 3D卫星环境可视化

- **地球**: 蓝色半透明球体
- **卫星**: 红色圆点，标注卫星ID
- **用户**: 绿色三角形，标注用户ID
- **卫星间连接**: 蓝色虚线，表示卫星间的通信链路
- **用户-卫星服务连接**: 橙色点线，表示用户与当前服务卫星的连接
- **可见性连接**: 灰色虚线，表示用户与可见卫星的连接

### 2. 状态信息文件

每个可视化步骤都会生成对应的JSON状态文件，包含：

```json
{
  "episode": 1,
  "step": 50,
  "total_steps": 500,
  "env_id": 0,
  "world_step": 50,
  "current_time": "Time(year=2025, month=7, day=6, hour=12, minute=0, second=50.0)",
  "satellites": [
    {
      "id": 0,
      "comp_resource": 850,
      "service_users": [0],
      "instance_list": [0]
    }
  ],
  "users": [
    {
      "id": 0,
      "current_sat": 0,
      "service_status": true
    }
  ]
}
```

### 3. 性能摘要图表

包含四个子图：
- **奖励曲线**: 显示episode内的奖励变化
- **迁移统计**: 显示迁移动作的分布
- **服务成功率**: 显示用户服务的成功情况
- **资源利用率**: 显示卫星资源的利用情况

## 配置选项

### 1. 可视化开关

```python
# 启用可视化
--enable_visualization

# 或在代码中设置
self.all_args.enable_visualization = True
```

### 2. 保存间隔

```python
# 修改可视化保存间隔
viz_save_interval = 50  # 每50步保存一次
```

### 3. 摘要生成间隔

```python
# 修改摘要生成间隔
if episode % 10 == 0:  # 每10个episode生成一次摘要
```

## 性能考虑

### 1. 存储空间

- HTML文件较大（约1-5MB每个）
- PNG文件较小（约100-500KB每个）
- 建议定期清理旧的可视化文件

### 2. 计算开销

- 可视化会增加训练时间约5-10%
- 可以通过调整保存间隔来平衡性能和可视化需求

### 3. 内存使用

- RGB数组模式会占用额外内存
- 建议在内存受限的环境中使用PNG模式

## 故障排除

### 1. 可视化文件未生成

- 检查是否启用了可视化功能
- 检查文件权限和磁盘空间
- 查看日志文件中的错误信息

### 2. HTML文件无法打开

- 确保浏览器支持WebGL
- 检查文件路径是否正确
- 尝试使用不同的浏览器

### 3. 性能问题

- 减少可视化保存间隔
- 使用PNG模式替代HTML模式
- 在训练完成后进行可视化

## 扩展功能

### 1. 自定义可视化

可以修改`plot_step_positions_interactive`函数来自定义可视化内容：

```python
def plot_step_positions_interactive(self, step_idx, save_dir):
    # 添加自定义的可视化元素
    # 例如：显示迁移路径、资源使用情况等
    pass
```

### 2. 动画生成

可以使用RGB数组模式生成训练过程的动画：

```python
import imageio

frames = []
for step in range(num_steps):
    rgb_array = env.render(mode='rgb_array')
    frames.append(rgb_array)

imageio.mimsave('training_animation.gif', frames, duration=0.5)
```

### 3. 实时监控

可以结合WebSocket等技术实现实时可视化监控：

```python
# 示例：实时发送可视化数据
import websocket

def send_visualization_data(data):
    ws = websocket.create_connection("ws://localhost:8080")
    ws.send(json.dumps(data))
    ws.close()
```

## 总结

通过使用这些可视化功能，你可以：

1. **直观理解**卫星环境的运行状态
2. **分析迁移策略**的效果和模式
3. **监控训练过程**中的关键指标
4. **生成报告**和演示材料
5. **调试问题**和优化算法

建议根据具体需求选择合适的可视化模式和保存间隔，以平衡性能和可视化效果。 