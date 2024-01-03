from . import models

MODEL_REGISTRY = {
    'MLP': models.MLP,
    'CNN2P2': models.CNN2P2,
    'AlexNet': models.AlexNet,
}

def get_model(model_name, build=False, **kwargs):
    """
    lookup model constructor from model registry by name

    if build=True, uses kwargs to build model and returns a model object
    otherwise just returns the constructor
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model ({model_name}) is not in MODEL_REGISTRY")
    model = MODEL_REGISTRY[model_name]
    if build:
        return model(**kwargs)
    return model