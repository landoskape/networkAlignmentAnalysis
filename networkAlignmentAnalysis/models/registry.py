from . import models

MODEL_REGISTRY = {
    'MLP': models.MLP,
    'CNN2P2': models.CNN2P2,
    'AlexNet': models.AlexNet,
}

def get_model(model_name):
    """lookup model constructor from model registry by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model ({model_name}) is not in MODEL_REGISTRY")
    return MODEL_REGISTRY[model_name]