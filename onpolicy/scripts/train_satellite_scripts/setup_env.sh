#!/bin/sh
# 卫星环境训练的环境设置脚本

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置项目根目录（假设脚本在 onpolicy/scripts/train_satellite_scripts/ 目录下）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== 卫星环境训练环境设置 ==="
echo "脚本目录: $SCRIPT_DIR"
echo "项目根目录: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "================================"

# 验证Python路径设置
python3 -c "import onpolicy; print('✓ onpolicy模块导入成功')" 2>/dev/null || {
    echo "✗ onpolicy模块导入失败，请检查路径设置"
    exit 1
} 