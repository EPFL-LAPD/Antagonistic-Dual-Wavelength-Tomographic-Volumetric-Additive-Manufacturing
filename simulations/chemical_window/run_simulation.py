"""Reproduce the selected chemical-window comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def scale_for_target_coverage(
    field: np.ndarray,
    target: np.ndarray,
    coverage: float,
    required_value: float,
) -> np.ndarray:
    reference = float(np.quantile(field[target], 1.0 - coverage))
    return field * (required_value / reference)


def material_state(
    field: np.ndarray, target: np.ndarray, vial: np.ndarray, required_value: float
) -> np.ndarray:
    above_gelation = field >= 1.0
    above_required = field >= required_value
    state = np.full(field.shape, -1, dtype=np.int8)
    state[vial & ~above_gelation] = 0
    state[target & above_gelation & ~above_required] = 1
    state[target & above_required] = 2
    state[vial & ~target & above_gelation] = 3
    return state


def main() -> None:
    parameters = json.loads((ROOT / "parameters.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "fields.npz") as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    target = data["target"].astype(bool)
    vial = data["vial"].astype(bool)
    negative = data["selected_negative_feature"].astype(bool)
    effective = np.maximum(
        data["antagonistic_activation"] - data["inhibition"], 0.0
    )
    activation = scale_for_target_coverage(
        data["activation_only"],
        target,
        parameters["target_coverage_fraction"],
        parameters["required_target_value"],
    )
    antagonistic = scale_for_target_coverage(
        effective,
        target,
        parameters["target_coverage_fraction"],
        parameters["required_target_value"],
    )

    metrics = {}
    for name, field in (
        ("activation_only", activation),
        ("antagonistic", antagonistic),
    ):
        metrics[name] = {
            "target_fraction_at_or_above_required_percent": float(
                100.0 * np.mean(field[target] >= parameters["required_target_value"])
            ),
            "target_fraction_at_or_above_gelation_percent": float(
                100.0 * np.mean(field[target] >= 1.0)
            ),
            "selected_negative_feature_fraction_above_gelation_percent": float(
                100.0 * np.mean(field[negative] >= 1.0)
            ),
        }

    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT / "result.npz",
        activation_only=activation,
        antagonistic_effective=antagonistic,
        activation_only_state=material_state(
            activation, target, vial, parameters["required_target_value"]
        ),
        antagonistic_state=material_state(
            antagonistic, target, vial, parameters["required_target_value"]
        ),
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
