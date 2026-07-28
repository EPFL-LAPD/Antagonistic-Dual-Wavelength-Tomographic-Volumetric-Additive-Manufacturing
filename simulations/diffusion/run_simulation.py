"""Compare single pre-generation with periodic inhibitor photogeneration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def no_flux_laplacian(vial: np.ndarray, pixel_size_mm: float) -> csr_matrix:
    index = np.full(vial.shape, -1, dtype=np.int64)
    index[vial] = np.arange(int(vial.sum()), dtype=np.int64)
    rows, columns, values = [], [], []
    ny, nx = vial.shape
    for y, x in np.argwhere(vial):
        centre = int(index[y, x])
        degree = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and vial[yy, xx]:
                rows.append(centre)
                columns.append(int(index[yy, xx]))
                values.append(1.0)
                degree += 1
        rows.append(centre)
        columns.append(centre)
        values.append(-float(degree))
    return coo_matrix(
        (np.asarray(values), (rows, columns)),
        shape=(int(vial.sum()), int(vial.sum())),
    ).tocsr() / pixel_size_mm**2


def simulate(
    activation: np.ndarray,
    inhibitor_generation: np.ndarray,
    vial: np.ndarray,
    laplacian: csr_matrix,
    parameters: dict,
    active_slots: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    cycles = parameters["cycles"]
    substeps = parameters["temporal_substeps_per_frame"]
    dt = parameters["frame_duration_s"] / substeps
    diffusion = parameters["diffusion_coefficient_mm2_s"]
    inhibitor_scale = 1.0 / len(active_slots)
    active_inhibitor = np.zeros(int(vial.sum()), dtype=np.float64)
    response = np.zeros_like(active_inhibitor)

    def diffuse_half(values: np.ndarray) -> np.ndarray:
        updated = values + 0.5 * dt * diffusion * (laplacian @ values)
        if float(updated.min()) < -1e-12:
            raise RuntimeError("The diffusion step produced a negative value.")
        updated[np.abs(updated) < 1e-14] = 0.0
        return updated

    for cycle in range(cycles):
        for increment in inhibitor_generation:
            local = (
                increment[vial] * inhibitor_scale
                if cycle in active_slots
                else np.zeros_like(active_inhibitor)
            )
            for _ in range(substeps):
                active_inhibitor += local / substeps
                active_inhibitor = diffuse_half(diffuse_half(active_inhibitor))
        for increment in activation:
            local = increment[vial] / cycles
            for _ in range(substeps):
                active_inhibitor = diffuse_half(active_inhibitor)
                delivered = local / substeps
                consumed = np.minimum(delivered, active_inhibitor)
                active_inhibitor -= consumed
                response += delivered - consumed
                active_inhibitor = diffuse_half(active_inhibitor)

    response_field = np.zeros(vial.shape, dtype=np.float64)
    inhibitor_field = np.zeros(vial.shape, dtype=np.float64)
    response_field[vial] = response
    inhibitor_field[vial] = active_inhibitor
    return response_field, inhibitor_field


def outcome_metrics(
    response: np.ndarray,
    target: np.ndarray,
    vial: np.ndarray,
    threshold: float,
    pixel_size_mm: float,
) -> dict[str, float]:
    above = (response >= threshold) & vial
    true_positive = int(np.count_nonzero(above & target))
    false_positive = int(np.count_nonzero(above & vial & ~target))
    false_negative = int(np.count_nonzero(target & ~above))
    return {
        "target_fraction_above_threshold": true_positive / int(target.sum()),
        "non_target_area_above_threshold_mm2": false_positive * pixel_size_mm**2,
        "iou": true_positive
        / (true_positive + false_positive + false_negative),
    }


def main() -> None:
    parameters = json.loads((ROOT / "parameters.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "increments_and_masks.npz") as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    vial = data["vial"].astype(bool)
    target = data["target"].astype(bool)
    pixel_size = parameters["pixel_size_mm"]
    laplacian = no_flux_laplacian(vial, pixel_size)
    single_slots = {
        value - 1 for value in parameters["single_pre_generation_slots_one_based"]
    }
    periodic_slots = {
        value - 1 for value in parameters["periodic_photogeneration_slots_one_based"]
    }
    single_response, single_inhibitor = simulate(
        data["activation_increments"],
        data["inhibitor_generation_increments"],
        vial,
        laplacian,
        parameters,
        single_slots,
    )
    periodic_response, periodic_inhibitor = simulate(
        data["activation_increments"],
        data["inhibitor_generation_increments"],
        vial,
        laplacian,
        parameters,
        periodic_slots,
    )

    metrics = {
        "single_pre_generation": outcome_metrics(
            single_response,
            target,
            vial,
            parameters["model_threshold"],
            pixel_size,
        ),
        "periodic_photogeneration": outcome_metrics(
            periodic_response,
            target,
            vial,
            parameters["model_threshold"],
            pixel_size,
        ),
    }
    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT / "result.npz",
        single_pre_generation_response=single_response,
        periodic_photogeneration_response=periodic_response,
        single_pre_generation_active_inhibitor=single_inhibitor,
        periodic_photogeneration_active_inhibitor=periodic_inhibitor,
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
