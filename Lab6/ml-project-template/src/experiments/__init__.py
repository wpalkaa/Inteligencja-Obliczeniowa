"""Experiment registry and experiment implementations."""

from src.experiments.batchnorm_experiment import BatchNormMLPExperiment
from src.experiments.linear_experiment import LinearClassifierExperiment
from src.experiments.mlp_experiment import MLPClassifierExperiment
from src.experiments.registry import ExperimentFactory, register_experiment

__all__ = [
    "BatchNormMLPExperiment",
    "ExperimentFactory",
    "LinearClassifierExperiment",
    "MLPClassifierExperiment",
    "register_experiment",
]
