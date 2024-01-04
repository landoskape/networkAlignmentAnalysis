from .alignment_comparison import AlignmentComparison

EXPERIMENT_REGISTRY = {
    'alignment_comparison': AlignmentComparison,
}

def get_experiment(experiment_name, build=False):
    """
    lookup model constructor from model registry by name

    if build=True, builds experiment and returns an experiment object
    otherwise just returns the constructor
    """
    if experiment_name not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment ({experiment_name}) is not in EXPERIMENT_REGISTRY")
    experiment = EXPERIMENT_REGISTRY[experiment_name]
    if build:
        return experiment()
    return experiment