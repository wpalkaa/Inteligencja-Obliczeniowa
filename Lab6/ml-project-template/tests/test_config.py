from pathlib import Path

import pytest

from src.experiments.registry import ExperimentFactory
from src.utils.config import ProjectConfig


def test_project_config_loads_yaml() -> None:
    config = ProjectConfig.from_yaml(Path("configs/iris_mlp_zscore.yaml"))
    assert config.project_name == "iris-classification-template"
    assert config.experiment_name == "iris_mlp_zscore"
    assert config.paths.raw_csv == Path("data/raw/iris.csv")
    assert len(config.data.feature_columns) == 4


def test_wrong_model_name() -> None:
    config = ProjectConfig.from_yaml(Path("configs/iris_mlp_zscore.yaml"))

    with pytest.raises(ValueError) as err:
        ExperimentFactory.build("brak modelu", config)
    assert "Nieznany eksperyment: brak modelu" in str(err.value)
