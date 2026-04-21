from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetArtifacts:
    frame: pd.DataFrame
    metadata: dict[str, object]


def build_diagnosis_dataframe(csv_path: str | Path = "data/raw/diagnosis.csv") -> DatasetArtifacts:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {path.absolute()}")

    frame = pd.read_csv(path)

    target_column = "diagnosis"

    if target_column in frame.columns:
        frame[target_column] = frame[target_column].replace({0: "zdrowy", 1: "chory"})
        feature_names = [col for col in frame.columns if col != target_column]
        target_names = frame[target_column].unique().tolist()
    else:
        feature_names = list(frame.columns)
        target_names = []

    metadata: dict[str, object] = {
        "source": f"local_file:{path.name}",
        "rows": int(frame.shape[0]),
        "columns": list(frame.columns),
        "target_names": [str(name) for name in target_names],
        "feature_names": feature_names,
        "description": "Diagnosis.csv",
    }

    return DatasetArtifacts(frame=frame, metadata=metadata)


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
