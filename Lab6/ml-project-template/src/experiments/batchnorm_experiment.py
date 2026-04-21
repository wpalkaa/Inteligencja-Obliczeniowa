from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.mlp_batchnorm import IrisBatchNormMLP


@register_experiment("mlp_batchnorm")
class BatchNormMLPExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "mlp_batchnorm"

    def build_model(self) -> IrisBatchNormMLP:
        input_dim = len(self.config.data.feature_columns)
        num_classes = len(self.config.data.class_names)

        return IrisBatchNormMLP(
            input_dim=input_dim,
            hidden_dims=self.config.training.hidden_dims,
            num_classes=num_classes,
        )
