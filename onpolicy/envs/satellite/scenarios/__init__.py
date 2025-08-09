import importlib.util
import os.path

def load(name):
    """
    动态加载场景模块
    Args:
        name: 场景文件名（如 'satellite_scenario.py'）
    Returns:
        加载的模块对象
    """
    pathname = os.path.join(os.path.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location('', pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
