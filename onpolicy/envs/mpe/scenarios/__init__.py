import importlib.util
import os.path

# 原来的imp只能兼容python3.4以下版本
# import imp
# import os.path as osp
# def load(name):
#     pathname = osp.join(osp.dirname(__file__), name)
#     return imp.load_source('', pathname)

def load(name):
    pathname = os.path.join(os.path.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location('', pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

