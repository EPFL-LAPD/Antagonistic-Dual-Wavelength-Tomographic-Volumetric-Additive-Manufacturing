"""Reproduce the selected attenuation comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def scale_for_target_coverage(
    field: np.ndarray, target: np.ndarray, coverage: float
) -> tuple[np.ndarray, float]:
    reference = float(np.quantile(field[target], 1.0 - coverage))
    return field / reference, reference


def intersection_over_union(
    predicted: np.ndarray, target: np.ndarray, region: np.ndarray
) -> float:
    union = np.count_nonzero((predicted | target) & region)
    return float(np.count_nonzero(predicted & target & region) / union)


def metrics(
    field: np.ndarray, data: dict[str, np.ndarray], threshold: float
) -> dict[str, float | int]:
    target = data["target"].astype(bool)
    vial = data["vial"].astype(bool)
    tooth_gaps = data["tooth_gaps"].astype(bool)
    internal_holes = data["internal_holes"].astype(bool)
    selected_negative = tooth_gaps | internal_holes
    tooth_profile = data["tooth_profile"].astype(bool)
    non_target = vial & ~target
    predicted = (field >= threshold) & vial
    connected = ndimage.label(
        predicted & target, structure=np.ones((3, 3), dtype=np.uint8)
    )[1]
    return {
        "target_fraction_above_model_threshold_percent": float(
            100.0 * np.mean(predicted[target])
        ),
        "tooth_gap_fraction_above_model_threshold_percent": float(
            100.0 * np.mean(predicted[tooth_gaps])
        ),
        "internal_hole_fraction_above_model_threshold_percent": float(
            100.0 * np.mean(predicted[internal_holes])
        ),
        "selected_negative_feature_fraction_above_model_threshold_percent": float(
            100.0 * np.mean(predicted[selected_negative])
        ),
        "non_target_vial_fraction_above_model_threshold_percent": float(
            100.0 * np.mean(predicted[non_target])
        ),
        "tooth_profile_iou": intersection_over_union(
            predicted, target, tooth_profile
        ),
        "global_iou": intersection_over_union(predicted, target, vial),
        "target_connected": int(connected == 1),
    }


def main() -> None:
    parameters = json.loads((ROOT / "parameters.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "fields.npz") as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    effective_raw = np.maximum(
        data["antagonistic_activation"] - data["inhibition"], 0.0
    )
    activation_only, _ = scale_for_target_coverage(
        data["activation_only"],
        data["target"].astype(bool),
        parameters["target_coverage_fraction"],
    )
    effective, effective_reference = scale_for_target_coverage(
        effective_raw,
        data["target"].astype(bool),
        parameters["target_coverage_fraction"],
    )
    antagonistic_activation = data["antagonistic_activation"] / effective_reference
    inhibition = data["inhibition"] / effective_reference

    summary = {
        "activation_only": metrics(
            activation_only, data, parameters["model_threshold"]
        ),
        "antagonistic": metrics(effective, data, parameters["model_threshold"]),
    }
    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT / "result.npz",
        activation_only=activation_only,
        antagonistic_activation=antagonistic_activation,
        inhibition=inhibition,
        effective=effective,
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
