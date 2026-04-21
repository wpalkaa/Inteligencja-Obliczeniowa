from __future__ import annotations

import argparse

import src.experiments  # noqa: F401
from src.experiments.registry import ExperimentFactory
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wypisz spis wszystkich dostępnych eksperymentow.")
    return parser.parse_args()


def main() -> int:
    _ = parse_args()
    logger = get_logger("list_experiments")

    available_experiments = ExperimentFactory.list()

    logger.info("Eksperymenty w rejestrze: %d.", len(available_experiments))
    for n in available_experiments:
        print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
