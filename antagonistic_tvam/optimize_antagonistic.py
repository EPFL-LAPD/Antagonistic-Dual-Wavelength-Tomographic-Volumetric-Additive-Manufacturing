"""Optimise 365 nm inhibition patterns against fixed 405 nm activation patterns.

Only the inhibition projector data are trainable. The loss is evaluated on
max(alpha D_405 - D_365, 0), where alpha is set by --activation-scale.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from tqdm import trange

from drtvam.geometry import geometries
from drtvam.loss import losses
from drtvam.utils import discretize, save_img, save_vol


PATTERN_KEY = "patterns"


def load_scene(config):
    """Load a single scene from configuration."""
    for key in ['target', 'vial', 'projector', 'sensor']:
        if key not in config:
            raise ValueError(f"Missing field '{key}' in the configuration file.")

    # Load vial geometry
    if 'type' not in config['vial']:
        raise ValueError("The vial geometry must have a 'type' field.")
    if config['vial']['type'] not in geometries.keys():
        raise ValueError(f"Unknown vial geometry: '{config['vial']['type']}'")

    vial = geometries[config['vial']['type']](config['vial'])

    if 'filename' not in config['target']:
        raise ValueError("Missing field 'filename' for the target shape.")

    # Target mesh transform
    mesh_type = os.path.splitext(config['target']['filename'])[1][1:]
    bbox = mi.load_dict({
        'type': mesh_type,
        'filename': config['target']['filename']
    }).bbox()

    c = 0.5 * (bbox.min + bbox.max)
    size = config['target'].get('size', 1.)
    center_pos_x = config['target'].get('box_center_x', 0.)
    center_pos_y = config['target'].get('box_center_y', 0.)
    center_pos_z = config['target'].get('box_center_z', 0.)
    scale_and_center = config['target'].get('scale_and_center', True)

    center_pos = mi.ScalarPoint3f(center_pos_x, center_pos_y, center_pos_z)

    if scale_and_center:
        target_to_world = mi.ScalarTransform4f().translate(center_pos) @ \
            mi.ScalarTransform4f().scale(size / dr.max(bbox.extents())) @ mi.ScalarTransform4f().translate(-c)
    else:
        target_to_world = mi.ScalarTransform4f()

    def get_sensor_transform(sensor_dict):
        sensor_scalex = sensor_dict.pop('scalex', 1.)
        sensor_scaley = sensor_dict.pop('scaley', 1.)
        sensor_scalez = sensor_dict.pop('scalez', 1.)
        return mi.ScalarTransform4f().scale(mi.ScalarPoint3f(sensor_scalex, sensor_scaley, sensor_scalez))

    sensor_to_world = get_sensor_transform(config['sensor'])

    # Create Mitsuba scene
    scene_dict = {
        'type': 'scene',
        'projector': config['projector'],
        'sensor': config['sensor'] | {'to_world': sensor_to_world},
        'target': {
            'type': mesh_type,
            'filename': config['target']['filename'],
            'to_world': target_to_world,
            'bsdf': {
                'type': 'null'
            }
        },
    } | vial.to_dict()

    if 'final_sensor' in config.keys():
        final_sensor_to_world = get_sensor_transform(config['final_sensor'])
        scene_dict['final_sensor'] = config['final_sensor'] | {'to_world': final_sensor_to_world}

    return scene_dict


def setup_scene_components(config, scene, params, output_prefix, patterns_fwd=None):
    """
    Setup scene components including filtering, sensors, and integrators.
    Returns a dictionary with all necessary components for optimization.
    """
    spp = config.get('spp', 4)
    spp_ref = config.get('spp_ref', 16)
    spp_grad = config.get('spp_grad', spp)
    max_depth = config.get('max_depth', 6)
    rr_depth = config.get('rr_depth', 6)
    time = config.get('time', 1.)
    progressive = config.get('progressive', False)
    transmission_only = config.get('transmission_only', True)
    regular_sampling = config.get('regular_sampling', False)
    filter_radon = config.get('filter_radon', False)

    patterns_key = 'projector.active_data'

    sensor = None
    final_sensor = None
    for s in scene.sensors():
        if s.id() == 'sensor':
            sensor = s
        elif s.id() == 'final_sensor':
            final_sensor = s

    if final_sensor is None:
        final_sensor = sensor
    if final_sensor.film().surface_aware:
        raise ValueError("The final sensor must not be surface-aware.")

    surface_aware = sensor.film().surface_aware

    integrator = mi.load_dict({
        'type': 'volumeintegrator',
        'max_depth': 3 if progressive else max_depth,
        'rr_depth': rr_depth,
        'print_time': time,
        'transmission_only': transmission_only,
        'regular_sampling': regular_sampling
    })

    # Computing reference
    if surface_aware:
        target = sensor.compute_volume(scene)
        save_vol(target[..., 0, None], os.path.join(output_prefix, "target_in.exr"))
        save_vol(target[..., 1, None], os.path.join(output_prefix, "target_out.exr"))
    else:
        target = discretize(scene, sensor=sensor)
        save_vol(target, os.path.join(output_prefix, "target.exr"))

    np.save(os.path.join(output_prefix, "target.npy"), target.numpy())

    # Filter Radon
    if filter_radon and patterns_fwd is None:
        radon_integrator = mi.load_dict({
            'type': 'radon',
            'max_depth': max_depth,
            'rr_depth': rr_depth,
            'print_time': time,
            'transmission_only': transmission_only
        })
        radon = mi.render(scene, integrator=radon_integrator, spp=config.get('spp_filter_radon', 4))

        active_pixels = dr.compress(radon.array > 0.) + dr.opaque(mi.UInt32, 0)
        dr.eval(active_pixels)

        if len(active_pixels) == 0:
            raise ValueError("No active pixels found in the Radon transform.")

        params['projector.active_pixels'] = active_pixels
        params[patterns_key] = dr.zeros(mi.Float, dr.width(active_pixels))
        params.update()

        del radon, radon_integrator
        dr.flush_malloc_cache()
        dr.sync_thread()

    # Filter corner
    if 'filter_corner' in config and patterns_fwd is None:
        corner_integrator = mi.load_dict({
            'type': 'corner',
            'regular_sampling': True,
        } | config['filter_corner'])
        corner = mi.render(scene, integrator=corner_integrator, spp=1)

        active_pixels = dr.compress(corner.array > 0.) + dr.opaque(mi.UInt32, 0)
        dr.eval(active_pixels)

        if len(active_pixels) == 0:
            raise ValueError("No active pixels found in the corner filter.")

        params['projector.active_pixels'] = active_pixels
        params[patterns_key] = dr.zeros(mi.Float, dr.width(active_pixels))
        params.update()

        del corner, corner_integrator
        dr.flush_malloc_cache()
        dr.sync_thread()

    # If not surface-aware, move target away
    if not surface_aware:
        params['target.vertex_positions'] += 1e5
        params.update()

    integrator_final = mi.load_dict({
        'type': 'volumeintegrator',
        'max_depth': config.get('max_depth_ref', 16),
        'rr_depth': config.get('rr_depth_ref', 8),
        'transmission_only': transmission_only,
        'regular_sampling': regular_sampling,
        'print_time': time
    })

    return {
        'scene': scene,
        'params': params,
        'sensor': sensor,
        'final_sensor': final_sensor,
        'integrator': integrator,
        'integrator_final': integrator_final,
        'target': target,
        'surface_aware': surface_aware,
        'patterns_key': patterns_key,
        'spp': spp,
        'spp_ref': spp_ref,
        'spp_grad': spp_grad,
        'max_depth': max_depth,
        'progressive': progressive,
    }


def optimize_fixed_activation(
    config_poly, config_inhib, patterns_fwd_poly, poly_scale=1.0
):
    """Optimise inhibition patterns while holding activation patterns fixed."""
    output = config_poly['output']

    # Create output directories
    os.makedirs(os.path.join(output, "activation"), exist_ok=True)
    os.makedirs(os.path.join(output, "inhibition"), exist_ok=True)
    os.makedirs(os.path.join(output, "activation", "patterns"), exist_ok=True)
    os.makedirs(os.path.join(output, "inhibition", "patterns"), exist_ok=True)

    # Activation scene
    print("Setting up activation scene...")
    scene_dict_poly = load_scene(config_poly)
    scene_poly = mi.load_dict(scene_dict_poly)
    params_poly = mi.traverse(scene_poly)

    components_poly = setup_scene_components(
        config_poly, scene_poly, params_poly,
        os.path.join(output, "activation"),
        patterns_fwd_poly
    )

    # Inhibition scene
    print("Setting up inhibition scene...")
    scene_dict_inhib = load_scene(config_inhib)
    scene_inhib = mi.load_dict(scene_dict_inhib)
    params_inhib = mi.traverse(scene_inhib)

    components_inhib = setup_scene_components(
        config_inhib, scene_inhib, params_inhib,
        os.path.join(output, "inhibition"),
        None
    )

    # Threshold loss
    if "loss" not in config_poly.keys():
        print("No loss function specified. Using thresholded loss.")
        config_poly['loss'] = {'type': 'threshold'}

    loss_type = config_poly['loss'].pop('type')
    if loss_type not in losses.keys():
        raise ValueError(f"Unknown loss type: '{loss_type}'. Available losses are: {list(losses.keys())}")

    loss_fn = losses[loss_type](config_poly['loss'])

    # Inhibition optimizer
    patterns_key_inhib = components_inhib['patterns_key']

    optimizer_config = dict(
        config_inhib.get("optimizer", {"type": "adam", "lr": 0.02})
    )
    optimizer_type = optimizer_config.pop("type")
    if optimizer_type != "adam":
        raise ValueError("The antagonistic workflow uses the Adam optimiser.")
    opt_inhib = mi.ad.Adam(**optimizer_config)

    # Only the inhibition pattern data are trainable.
    opt_inhib[patterns_key_inhib] = params_inhib[patterns_key_inhib]

    n_steps = config_inhib.get('n_steps', 40)
    params_poly['projector.active_data'] = poly_scale * patterns_fwd_poly.flatten()
    params_poly.update()

    print("Optimizing inhibition patterns against fixed activation...")

    target = components_poly['target']

    for i in trange(n_steps):
        # Handle progressive rendering
        if components_poly['progressive'] and i == 5:
            components_poly['integrator'].max_depth = components_poly['max_depth']
        if components_inhib['progressive'] and i == 5:
            components_inhib['integrator'].max_depth = components_inhib['max_depth']

        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
            params_inhib.update(opt_inhib)

            vol_polymerize = mi.render(
                    scene_poly, params_poly,
                    integrator=components_poly['integrator'],
                    sensor=components_poly['sensor'],
                    spp=components_poly['spp'],
                    spp_grad=components_poly['spp_grad'],
                    seed=i
            )
            dr.schedule(vol_polymerize)

            vol_inhibit = mi.render(
                    scene_inhib, params_inhib,
                    integrator=components_inhib['integrator'],
                    sensor=components_inhib['sensor'],
                    spp=components_inhib['spp'],
                    spp_grad=components_inhib['spp_grad'],
                    seed=i + 1000000  # Different seed
            )
            dr.schedule(vol_inhibit)

            vol_effective = dr.maximum(0, vol_polymerize - vol_inhibit)
            loss = loss_fn(vol_effective, target, params_inhib['projector.active_data'])

            dr.eval(loss)
            dr.backward(loss)

            if dr.all(loss == 0):
                print("Converged")
                break

            opt_inhib.step()
            opt_inhib[patterns_key_inhib] = dr.maximum(
                dr.detach(opt_inhib[patterns_key_inhib]), 0
            )

    params_inhib.update(opt_inhib)

    # Final reconstructions
    print("Rendering final activation state...")
    vol_final_poly = mi.render(
        scene_poly, params_poly,
        spp=components_poly['spp_ref'],
        integrator=components_poly['integrator_final'],
        sensor=components_poly['final_sensor']
    )

    print("Rendering final inhibition state...")
    vol_final_inhib = mi.render(
        scene_inhib, params_inhib,
        spp=components_inhib['spp_ref'],
        integrator=components_inhib['integrator_final'],
        sensor=components_inhib['final_sensor']
    )

    # Activation outputs
    output_poly = os.path.join(output, "activation")
    np.save(os.path.join(output_poly, "final.npy"), vol_final_poly.numpy())
    save_vol(vol_final_poly, os.path.join(output_poly, "final.exr"))

    imgs_final_poly = scene_poly.emitters()[0].patterns()
    dr.eval(imgs_final_poly)

    print("Saving activation patterns...")
    for idx in trange(imgs_final_poly.shape[0]):
        save_img(imgs_final_poly[idx], os.path.join(output_poly, "patterns", f"{idx:04d}.exr"))
    np.savez_compressed(os.path.join(output_poly, "patterns.npz"), patterns=imgs_final_poly.numpy())

    # Normalized uint8 patterns
    array_poly = imgs_final_poly.numpy()
    max_intensity_poly = np.max(array_poly)
    if max_intensity_poly > 0:
        normalized_poly = (array_poly / max_intensity_poly * 255).astype(np.uint8)
    else:
        normalized_poly = np.zeros_like(array_poly, dtype=np.uint8)
    np.savez_compressed(os.path.join(output_poly, "patterns_normalized_uint8.npz"), patterns=normalized_poly)

    # Inhibition outputs
    output_inhib = os.path.join(output, "inhibition")
    np.save(os.path.join(output_inhib, "final.npy"), vol_final_inhib.numpy())
    save_vol(vol_final_inhib, os.path.join(output_inhib, "final.exr"))

    imgs_final_inhib = scene_inhib.emitters()[0].patterns()
    dr.eval(imgs_final_inhib)

    print("Saving inhibition patterns...")
    for idx in trange(imgs_final_inhib.shape[0]):
        save_img(imgs_final_inhib[idx], os.path.join(output_inhib, "patterns", f"{idx:04d}.exr"))
    np.savez_compressed(os.path.join(output_inhib, "patterns.npz"), patterns=imgs_final_inhib.numpy())

    # Normalized uint8 patterns
    array_inhib = imgs_final_inhib.numpy()
    max_intensity_inhib = np.max(array_inhib)
    if max_intensity_inhib > 0:
        normalized_inhib = (array_inhib / max_intensity_inhib * 255).astype(np.uint8)
    else:
        normalized_inhib = np.zeros_like(array_inhib, dtype=np.uint8)
    np.savez_compressed(os.path.join(output_inhib, "patterns_normalized_uint8.npz"), patterns=normalized_inhib)

    vol_combined = vol_final_poly - vol_final_inhib
    vol_combined_clamped = dr.maximum(vol_combined, 0)
    np.save(os.path.join(output, "vol_combined_clamped.npy"), vol_combined_clamped.numpy())
    save_vol(vol_combined_clamped, os.path.join(output, "vol_combined_clamped.exr"))

    return vol_final_poly, vol_final_inhib, vol_combined_clamped


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def validate_domains(activation: dict, inhibition: dict) -> None:
    for key in ("vial", "projector", "sensor", "target"):
        if key not in activation or key not in inhibition:
            raise ValueError(f"Both configurations must define '{key}'.")

    activation_film = activation["sensor"]["film"]
    inhibition_film = inhibition["sensor"]["film"]
    film_keys = ("resx", "resy", "resz")
    scale_keys = ("scalex", "scaley", "scalez")
    if any(activation_film[k] != inhibition_film[k] for k in film_keys):
        raise ValueError("Activation and inhibition reconstruction grids differ.")
    if any(activation["sensor"].get(k, 1.0) != inhibition["sensor"].get(k, 1.0)
           for k in scale_keys):
        raise ValueError("Activation and inhibition reconstruction domains differ.")
    if activation["target"].get("size", 1.0) != inhibition["target"].get("size", 1.0):
        raise ValueError("Activation and inhibition target scales differ.")


def load_activation_patterns(path: Path, config: dict) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        if PATTERN_KEY not in archive.files:
            raise KeyError(f"{path} does not contain an array named '{PATTERN_KEY}'.")
        patterns = archive[PATTERN_KEY]
    projector = config["projector"]
    expected = (
        int(projector["n_patterns"]),
        int(projector["resy"]),
        int(projector["resx"]),
    )
    if patterns.shape != expected:
        raise ValueError(f"Activation stack shape {patterns.shape} does not match {expected}.")
    if not np.issubdtype(patterns.dtype, np.floating):
        raise TypeError("The fixed activation stack must use floating-point values.")
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a 365 nm inhibition stack against fixed 405 nm patterns."
    )
    parser.add_argument("--activation-config", type=Path, required=True)
    parser.add_argument("--inhibition-config", type=Path, required=True)
    parser.add_argument("--activation-patterns", type=Path, required=True)
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--backend", choices=("cuda", "llvm"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.activation_scale < 0:
        raise ValueError("--activation-scale must be non-negative.")

    activation_path = args.activation_config.resolve()
    inhibition_path = args.inhibition_config.resolve()
    patterns_path = args.activation_patterns.resolve()
    config_poly = load_config(activation_path)
    config_inhib = load_config(inhibition_path)
    validate_domains(config_poly, config_inhib)
    patterns_poly = load_activation_patterns(patterns_path, config_poly)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    config_poly["output"] = str(output)
    config_inhib["output"] = str(output)
    output.mkdir(parents=True, exist_ok=True)

    mi.set_variant(f"{args.backend}_ad_mono")
    mi.Thread.thread().file_resolver().append(str(activation_path.parent))
    mi.Thread.thread().file_resolver().append(str(inhibition_path.parent))

    with (output / "opt_config_activation.json").open("w", encoding="utf-8") as handle:
        json.dump(config_poly, handle, indent=2)
    with (output / "opt_config_inhibition.json").open("w", encoding="utf-8") as handle:
        json.dump(config_inhib, handle, indent=2)

    optimize_fixed_activation(
        config_poly,
        config_inhib,
        patterns_fwd_poly=patterns_poly,
        poly_scale=args.activation_scale,
    )


if __name__ == "__main__":
    main()
