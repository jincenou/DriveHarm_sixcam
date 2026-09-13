"""Thin adapter from DriveHarm jobs to the frozen train STORM renderer."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image

from .contracts import atomic_jsonl, canonical_sha256, iter_jsonl, sha256_file


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("L")) > 5


def _save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(path)


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8), mode="RGB").save(path)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return deterministic 8-connected components without another dependency."""
    remaining = mask.copy()
    result: list[np.ndarray] = []
    height, width = mask.shape
    while remaining.any():
        y, x = map(int, np.argwhere(remaining)[0])
        component = np.zeros_like(mask)
        queue = deque([(y, x)])
        remaining[y, x] = False
        while queue:
            cy, cx = queue.popleft()
            component[cy, cx] = True
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        queue.append((ny, nx))
        result.append(component)
    return result


def _foreground_receipt(
    row: dict[str, Any], destination: Path
) -> tuple[Path | None, dict[str, Any] | None]:
    layers = row["composition_layers"]
    full = _mask(Path(layers["inserted_full_alpha"]))
    visible = _mask(Path(layers["inserted_visible_alpha"]))
    exact = _mask(Path(layers["target_actor_sam2_alpha"]))
    renderer_selected = full & ~visible
    full_pixels = int(full.sum())
    if not renderer_selected.any():
        return None, None
    scene_path = Path(layers["removed_scene_depth_m"])
    asset_path = Path(layers["inserted_visible_depth_m"])
    scene_depth = np.load(scene_path, allow_pickle=False)
    asset_depth = np.load(asset_path, allow_pickle=False)
    if scene_depth.shape != full.shape or asset_depth.shape != full.shape:
        raise ValueError("legacy depth and alpha shapes differ")
    valid = (
        np.isfinite(scene_depth)
        & np.isfinite(asset_depth)
        & (scene_depth > 0)
        & (asset_depth > 0)
    )
    nearer = valid & (
        scene_depth + np.maximum(0.3, 0.02 * asset_depth) < asset_depth
    )
    selected = np.zeros_like(full)
    component_rows: list[dict[str, Any]] = []
    accepted_indices: list[int] = []
    for index, component in enumerate(_components(renderer_selected), 1):
        seeds = int((component & nearer).sum())
        accepted = seeds >= 4
        if accepted:
            selected |= component
            accepted_indices.append(index)
        component_rows.append(
            {
                "component_index": index,
                "component_pixels": int(component.sum()),
                "verified_nearer_seed_pixels": seeds,
                "accepted": accepted,
            }
        )
    minimum = max(16, int(np.ceil(0.04 * full_pixels)))
    if int(selected.sum()) < minimum:
        return None, None
    path = destination / "foreground_occlusion.png"
    _save_mask(path, selected)
    target_depth_values = asset_depth[
        full & np.isfinite(asset_depth) & (asset_depth > 0)
    ]
    if not target_depth_values.size:
        raise ValueError("asset depth has no positive finite support")
    target_depth = float(np.median(target_depth_values))
    static = {
        "ordinal": 0,
        "evidence_kind": "unannotated_static_depth_seed_region",
        "region_id": "legacy-renderer-scene-depth",
        "minimum_verified_seed_pixels_per_component": 4,
        "components": component_rows,
        "accepted_component_indices": accepted_indices,
        "selected_pixels": int(selected.sum()),
        "accepted": bool(selected.any()),
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "policy": "distinct_nearer_official_or_static_depth_v1",
        "target_instance_token": row["instance_token"],
        "target_center_depth_m": target_depth,
        "absolute_depth_margin_m": 0.3,
        "relative_depth_margin": 0.02,
        "minimum_overlap_pixels": minimum,
        "minimum_static_seed_pixels": 4,
        "image_shape_hw": list(full.shape),
        "full_asset_support_sha256": _array_sha256(full.astype(np.uint8)),
        "target_exact_support_sha256": _array_sha256(exact.astype(np.uint8)),
        "scene_depth_sha256": sha256_file(scene_path),
        "asset_depth_sha256": sha256_file(asset_path),
        "nearer_depth_support_sha256": _array_sha256(nearer.astype(np.uint8)),
        "official_instance_decisions": [],
        "static_region_decisions": [static],
        "full_projected_pixels": full_pixels,
        "selected_occlusion_pixels": int(selected.sum()),
        "selected_occlusion_fraction": int(selected.sum()) / max(full_pixels, 1),
        "selected_occlusion_mask_sha256": _array_sha256(selected.astype(np.uint8)),
        "target_exact_is_diagnostic_only": True,
        "target_exact_overlap_pixels": int((selected & exact).sum()),
        "quality_gate_pass": True,
        "independent_foreground_verified": bool(selected.any()),
        "renderer_verified_occlusion_pixels": int(renderer_selected.sum()),
        "restored_asset_pixels": int((renderer_selected & ~selected).sum()),
    }
    receipt["decision_sha256"] = canonical_sha256(receipt)
    return path, receipt


def _geometry(row: dict[str, Any], occluded: bool) -> dict[str, Any]:
    quality = row["quality"]
    renderer_geometry = quality.get("full_projection_geometry_gate") or {}
    bbox_key = (
        "visibility_aware_projection_bbox_vs_target_sam2"
        if occluded
        else "full_projection_bbox_vs_target_sam2"
    )
    silhouette_key = (
        "visibility_aware_projection_silhouette_vs_target_sam2"
        if occluded
        else "full_projection_silhouette_vs_target_sam2"
    )
    bbox = quality[bbox_key]
    silhouette = quality[silhouette_key]
    integrity = quality.get("inserted_appearance_vs_target") or {}
    return {
        # Preserve the train renderer's independently audited strict route.  The
        # numeric fields below remain the adaptive recovery route; neither is a
        # substitute for the hard safety checks consumed by compose.py.
        "renderer_quality_gate_pass": quality.get("quality_gate_pass") is True,
        "renderer_geometry_gate_pass": renderer_geometry.get("quality_gate_pass")
        is True,
        "target_actor_pixels": int(quality.get("target_actor_pixels") or 0),
        "complete_projected_asset_pixels": int(
            quality.get("complete_projected_asset_pixels") or 0
        ),
        "silhouette_iou": float(silhouette["iou"]),
        "center_error_px": float(bbox["center_error_px"]),
        "width_ratio": float(bbox["width_ratio"]),
        "height_ratio": float(bbox["height_ratio"]),
        "bottom_error_px": float(
            (quality.get("inserted_bottom_vs_target_sam2") or {})[
                "absolute_error_px"
            ]
        ),
        "orientation_error_deg": (
            quality.get("inserted_orientation_vs_target_sam2") or {}
        ).get("absolute_error_deg"),
        "ground_lock_pass": (quality.get("ground_lock") or {}).get(
            "quality_gate_pass"
        )
        is True,
        "catastrophic_asset_safety_pass": integrity.get(
            "catastrophic_asset_safety_pass"
        )
        is True,
        "broken_or_doubled_asset": False,
        "verified_foreground_occlusion_applied": occluded,
        "degenerate_rectangle_mask": (
            quality.get("complete_asset_visibility") or {}
        ).get("degenerate_rectangle_mask")
        is True,
    }


def _legacy_index(
    manifest: dict[str, Any],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for job in manifest.get("jobs") or []:
        context = str(job.get("multi_case_id") or "")
        if not context:
            raise ValueError("legacy render opportunity has no multi_case_id")
        for raw, usable in (job.get("target_usable_by_frame_camera") or {}).items():
            if not usable:
                continue
            frame, camera = raw.split(":c", 1)
            key = (context, job["obj_id"], int(frame), camera)
            if key in result:
                raise ValueError(f"ambiguous legacy render opportunity: {key}")
            result[key] = job
    return result


def _run_legacy(
    profile: dict[str, Any], job_indices: list[int], output_root: Path, gpu: int
) -> None:
    selection = output_root / "legacy_job_indices.txt"
    selection.write_text("".join(f"{value}\n" for value in job_indices), encoding="utf-8")
    command = [
        str(profile["python"]),
        "-m",
        "pipeline.render_batch",
        "--job-root",
        str(profile["job_root"]),
        "--output-root",
        str(output_root / "legacy"),
        "--data-root",
        str(profile["data_root"]),
        "--annotation-list",
        str(profile["annotation_list"]),
        "--raw-nuscenes-root",
        str(profile["raw_nuscenes_root"]),
        "--storm-checkpoint",
        str(profile["storm_checkpoint"]),
        "--cvac-checkpoint",
        str(profile["cvac_checkpoint"]),
        "--dcn-checkpoint",
        str(profile["dcn_checkpoint"]),
        "--gpus",
        str(gpu),
        "--workers-per-gpu",
        "1",
        "--only-jobs-file",
        str(selection),
        "--allow-strong-automated-identity-review",
        "--require-independent-occlusion-layers",
    ]
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PATH"] = os.pathsep.join(
        [str(Path(profile["python"]).resolve(strict=True).parent), environment.get("PATH", "")]
    )
    completed = subprocess.run(
        command, cwd=profile["repo_root"], env=environment, check=False
    )
    # The frozen batch controller returns one when a bounded subset of jobs
    # fails after writing per-job state.  Preserve those failures for group
    # quarantine; any other exit code is an orchestration failure.
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"legacy renderer failed systemically with exit {completed.returncode}"
        )


def _legacy_result_path(
    output_root: Path, legacy: dict[str, Any]
) -> Path:
    return (
        output_root
        / "legacy/single_asset_results"
        / f"job_{int(legacy['job_index']):04d}"
        / f"{legacy['scene_name']}__{legacy['obj_id']}"
        / "triplet_result.json"
    )


def adapt(args: argparse.Namespace) -> None:
    profile = _load(args.profile.resolve(strict=True))
    required_profile = (
        "python",
        "repo_root",
        "job_root",
        "data_root",
        "annotation_list",
        "raw_nuscenes_root",
        "storm_checkpoint",
        "cvac_checkpoint",
        "dcn_checkpoint",
    )
    missing_profile = [key for key in required_profile if not profile.get(key)]
    if missing_profile:
        raise ValueError(
            "renderer profile is missing required fields: "
            + ", ".join(missing_profile)
        )
    for key in required_profile:
        Path(str(profile[key])).resolve(strict=True)
    jobs = list(iter_jsonl(args.jobs.resolve(strict=True)))
    manifest = _load(Path(profile["job_root"]) / "single_asset_jobs.json")
    index = _legacy_index(manifest)
    bindings: dict[str, list[dict[str, Any]]] = {}
    selected_indices: set[int] = set()
    for job in jobs:
        context = str(job.get("context_id") or job.get("window_id") or "")
        if not context:
            raise ValueError(f"render job has no context identity: {job['sample_id']}")
        key_suffix = (int(job["frame_index"]), str(job["camera_id"]))
        matched = []
        for asset in job["assets"]:
            legacy = index.get((context, asset["obj_id"], *key_suffix))
            if legacy is None:
                raise ValueError(f"no legacy STORM opportunity for {job['sample_id']}:{asset['obj_id']}")
            binding = legacy["exact_asset_binding"]
            if (
                binding["instance_token"] != asset["instance_token"]
                or binding["asset_sha256"] != asset["asset_sha256"]
                or asset.get("forward_axis") != "+X"
            ):
                raise ValueError(f"legacy identity differs for {asset['obj_id']}")
            matched.append(legacy)
            selected_indices.add(int(legacy["job_index"]))
        bindings[job["sample_id"]] = matched
    physical_gpu = int(os.environ.get("DRIVEHARM_PHYSICAL_GPU", args.gpu))
    if args.reuse_legacy:
        missing = [
            str(_legacy_result_path(args.output_root, legacy))
            for rows in bindings.values()
            for legacy in rows
            if not _legacy_result_path(args.output_root, legacy).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "--reuse-legacy requires every frozen renderer result; missing: "
                + ", ".join(missing[:3])
            )
    else:
        _run_legacy(profile, sorted(selected_indices), args.output_root, physical_gpu)
    checkpoint_hashes = {
        name: sha256_file(Path(profile[f"{name}_checkpoint"]))
        for name in ("storm", "cvac", "dcn")
    }
    results: list[dict[str, Any]] = []
    for job in jobs:
        missing = [
            str(_legacy_result_path(args.output_root, legacy))
            for legacy in bindings[job["sample_id"]]
            if not _legacy_result_path(args.output_root, legacy).is_file()
        ]
        if missing:
            result = {
                "schema_version": 1,
                "status": "failed",
                "job_kind": job.get("job_kind"),
                "sample_id": job["sample_id"],
                "job_sha256": job["job_sha256"],
                "split": job.get("split"),
                "context_id": job.get("context_id"),
                "scene_name": job["scene_name"],
                "frame_index": job["frame_index"],
                "camera_id": str(job["camera_id"]),
                "selected_obj_ids": list(job["selected_obj_ids"]),
                "checkpoint_sha256": checkpoint_hashes,
                "error": {
                    "type": "LegacyRenderResultMissing",
                    "message": f"{len(missing)} bound legacy result(s) missing",
                    "examples": missing[:3],
                },
            }
            result["result_sha256"] = canonical_sha256(result)
            results.append(result)
            print(f"visible backfill {job['sample_id']} failed", flush=True)
            continue
        sample_root = args.output_root / "converted" / job["sample_id"]
        flat_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for legacy in bindings[job["sample_id"]]:
            result_path = _legacy_result_path(args.output_root, legacy)
            payload = _load(result_path)
            flat = next(
                row
                for row in payload["flat_records"]
                if int(row["frame_index"]) == int(job["frame_index"])
                and str(row["camera_id"]) == str(job["camera_id"])
            )
            flat_rows.append((legacy, payload, flat))
        target_hashes = {
            row[2]["content_hashes"]["target_png_sha256"] for row in flat_rows
        }
        gt_hashes = {
            row[2]["content_hashes"]["sensor_gt_png_sha256"] for row in flat_rows
        }
        if len(target_hashes) != 1 or len(gt_hashes) != 1:
            raise ValueError(f"multi-asset backgrounds differ: {job['sample_id']}")
        first = flat_rows[0][2]
        target_path = Path(first["paths"]["storm_baseline"])
        gt_path = Path(first["paths"]["real_gt"])
        target_rgb = np.asarray(Image.open(target_path).convert("RGB"))
        removed_rgb = target_rgb.copy()
        removal_union = np.zeros(target_rgb.shape[:2], dtype=bool)
        layers = []
        for ordinal, (legacy, payload, row) in enumerate(flat_rows):
            layer_root = sample_root / f"asset{ordinal:02d}"
            source_layers = row["composition_layers"]
            removal = _mask(Path(source_layers["sensor_local_edit_matte"]))
            replacement = np.asarray(Image.open(row["paths"]["removed"]).convert("RGB"))
            removed_rgb[removal] = replacement[removal]
            removal_union |= removal
            foreground_path, receipt = _foreground_receipt(row, layer_root)
            occluded = bool(receipt and receipt["selected_occlusion_pixels"])
            full_rgb = Path(source_layers["inserted_full_premultiplied_rgb"])
            full_alpha = Path(source_layers["inserted_full_alpha"])
            exact = Path(source_layers["target_actor_sam2_alpha"])
            official_dimensions = list(payload["asset"]["target_size_lwh_m"])
            expected_asset = next(
                asset for asset in job["assets"] if asset["obj_id"] == legacy["obj_id"]
            )
            if legacy.get("canonical_asset_sha256") != expected_asset["asset_sha256"]:
                raise ValueError(f"canonical asset hash differs for {legacy['obj_id']}")
            if not np.allclose(official_dimensions, expected_asset["official_dimensions_m"]):
                raise ValueError(f"official dimensions differ for {legacy['obj_id']}")
            content = {
                "rgb": sha256_file(full_rgb),
                "alpha": sha256_file(full_alpha),
                "target_exact_mask": sha256_file(exact),
            }
            if foreground_path:
                content["foreground_occlusion_mask"] = sha256_file(foreground_path)
            asset_depth = np.load(source_layers["inserted_visible_depth_m"], allow_pickle=False)
            positive = asset_depth[np.isfinite(asset_depth) & (asset_depth > 0)]
            if not positive.size:
                raise ValueError(f"asset depth is empty for {legacy['obj_id']}")
            layers.append(
                {
                    "obj_id": legacy["obj_id"],
                    "instance_token": legacy["instance_token"],
                    "asset_sha256": expected_asset["asset_sha256"],
                    "exact_identity_verified": True,
                    "rgb": str(full_rgb.resolve()),
                    "alpha": str(full_alpha.resolve()),
                    "official_dimensions_m": official_dimensions,
                    "forward_axis": "+X",
                    "official_dimensions_applied": True,
                    "canonical_orientation_applied": payload["asset"].get(
                        "canonical_transform_reapplied"
                    )
                    is False,
                    "bottom_ground_locked": (
                        payload.get("bounded_image_space_refinement") or {}
                    ).get("ground_lock", {}).get("maximum_bottom_error_m")
                    == 0.0,
                    "content_sha256": content,
                    "median_depth_m": float(np.median(positive)),
                    "quality": _geometry(row, occluded),
                    **(
                        {
                            "foreground_occlusion_mask": str(foreground_path.resolve()),
                            "foreground_occlusion_receipt": receipt,
                        }
                        if foreground_path
                        else {}
                    ),
                    "target_exact_mask": str(exact.resolve()),
                }
            )
        removed_path = sample_root / "actor_removed.png"
        removal_path = sample_root / "actor_removed_mask.png"
        _save_rgb(removed_path, removed_rgb)
        _save_mask(removal_path, removal_union)
        cleanup = {
            "schema_version": 1,
            "policy": "union_of_hash_bound_legacy_actor_cleanup_masks_v1",
            "observation_technical_usable": all(
                row[2]["quality"]["original_actor_cleanup"][
                    "observation_technical_usable"
                ]
                is True
                for row in flat_rows
            ),
            "quality_gate_pass": all(
                row[2]["quality"]["original_actor_cleanup"]["quality_gate_pass"]
                is True
                for row in flat_rows
            ),
            "source_decision_sha256": [
                row[2]["quality"]["original_actor_cleanup"]["decision_sha256"]
                for row in flat_rows
            ],
        }
        cleanup["decision_sha256"] = canonical_sha256(cleanup)
        result = {
            "schema_version": 1,
            "status": "complete",
            "sample_id": job["sample_id"],
            "job_sha256": job["job_sha256"],
            "scene_name": job["scene_name"],
            "frame_index": job["frame_index"],
            "camera_id": str(job["camera_id"]),
            "selected_obj_ids": list(job["selected_obj_ids"]),
            "checkpoint_sha256": checkpoint_hashes,
            "render_domain": "storm",
            "camera_exposure": {
                key: first["quality"][key]
                for key in (
                    "sample_data_timestamp_us",
                    "camera_exposure_timestamp_us",
                    "storm_normalized_time_seconds",
                )
            },
            "original_actor_cleanup_receipt": cleanup,
            "quality": {
                "target_storm_vs_sensor_before_insertion": first["quality"][
                    "target_storm_vs_sensor_before_insertion"
                ]
            },
            "paths": {
                "real_gt": str(gt_path.resolve()),
                "storm_baseline": str(target_path.resolve()),
                "actor_removed": str(removed_path.resolve()),
                "actor_removed_mask": str(removal_path.resolve()),
            },
            "source_sha256": {
                "real_gt": sha256_file(gt_path),
                "storm_baseline": sha256_file(target_path),
                "actor_removed": sha256_file(removed_path),
                "actor_removed_mask": sha256_file(removal_path),
            },
            "asset_layers": layers,
        }
        results.append(result)
        print(f"visible backfill {job['sample_id']} complete", flush=True)
    atomic_jsonl(args.results, results)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--jobs", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--results", type=Path, required=True)
    value.add_argument("--gpu", type=int, required=True)
    value.add_argument("--profile", type=Path, required=True)
    value.add_argument(
        "--reuse-legacy",
        action="store_true",
        help="rebuild adapter metadata from existing frozen renderer outputs",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    if args.gpu != 0:
        raise ValueError("DriveHarm exposes one GPU; adapter GPU must be 0")
    args.output_root.mkdir(parents=True, exist_ok=True)
    adapt(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
