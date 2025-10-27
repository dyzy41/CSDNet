
from .CSDNet.StCoNet import CSDNet


# 方法一：手动构建映射字典
_model_factory = {
    'CSDNet': CSDNet,
}

def build_model(name: str, backbone:str,  *args, **kwargs):
    """
    根据 name 字符串返回对应的模型实例。
    支持传入构造函数的参数 args, kwargs。
    """
    try:
        ModelClass = _model_factory[name]
    except KeyError:
        raise ValueError(f"Unknown model name '{name}'. Available: {list(_model_factory.keys())}")
    return ModelClass(backbone, *args, **kwargs)
