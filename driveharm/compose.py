"""STORM pair composition with independently verified foreground occlusion."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .contracts import (
    IMAGE_SIZE,
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    iter_jsonl,
    sha256_file,
)


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != IMAGE_SIZE:
            raise ValueError(f"invalid RGB image: {path}")
        return np.asarray(image, dtype=np.float32) / 255.0


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.size != IMAGE_SIZE:
            raise ValueError(f"invalid mask dimensions: {path}")
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        data = np.clip(value * 255.0, 0, 255).round().astype(np.uint8)
        Image.fromarray(data, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def geometry_thresholds(projected_pixels: int) -> dict[str, float]:
    if projected_pixels < 100:
        return {
            "iou": 0.30,
            "center": 12.0,
            "ratio_low": 0.70,
            "ratio_high": 1.40,
            "bottom": 10.0,
            "yaw": 25.0,
        }
    if projected_pixels < 400:
        return {
            "iou": 0.33,
            "center": 11.0,
            "ratio_low": 0.72,
            "ratio_high": 1.37,
            "bottom": 10.0,
            "yaw": 25.0,
        }
    if projected_pixels < 1600:
        return {
            "iou": 0.36,
            "center": 10.0,
            "ratio_low": 0.75,
            "ratio_high": 1.33,
            "bottom": 10.0,
            "yaw": 24.0,
        }
    return {
        "iou": 0.40,
        "center": 10.0,
        "ratio_low": 0.78,
        "ratio_high": 1.30,
        "bottom": 10.0,
        "yaw": 22.0,
    }


def geometry_pass(quality: dict[str, Any]) -> bool:
    def number(key: str, default: float) -> float:
        value = quality.get(key)
        return default if value is None else float(value)

    pixels = max(
        int(quality.get("target_actor_pixels") or 0),
        int(quality.get("complete_projected_asset_pixels") or 0),
    )
    limits = geometry_thresholds(pixels)
    iou = number("silhouette_iou", 0.0)
    center = number("center_error_px", math.inf)
    width = number("width_ratio", 0.0)
    height = number("height_ratio", 0.0)
    bottom = number("bottom_error_px", math.inf)
    yaw = quality.get("orientation_error_deg")
    occluded = quality.get("verified_foreground_occlusion_applied") is True
    hard_safety = bool(
        quality.get("ground_lock_pass") is True
        and quality.get("catastrophic_asset_safety_pass") is True
        and quality.get("broken_or_doubled_asset") is not True
        and (yaw is None or abs(float(yaw)) < 90.0)
    )
    strict_renderer_proof = bool(
        quality.get("renderer_quality_gate_pass") is True
        and quality.get("renderer_geometry_gate_pass") is True
    )
    adaptive_recovery = bool(
        iou >= limits["iou"]
        and center <= limits["center"]
        and limits["ratio_low"] <= width <= limits["ratio_high"]
        and limits["ratio_low"] <= height <= limits["ratio_high"]
        and (occluded or bottom <= limits["bottom"])
        and (yaw is None or float(yaw) <= limits["yaw"])
    )
    return hard_safety and (strict_renderer_proof or adaptive_recovery)


def edit_overlap_pass(masks: list[np.ndarray], maximum_fraction: float = 0.02) -> bool:
    for index, first in enumerate(masks):
        for second in masks[index + 1 :]:
            overlap = int((first & second).sum())
            smaller = min(int(first.sum()), int(second.sum()))
            if overlap / max(smaller, 1) > maximum_fraction:
                return False
    return True


def _verified_foreground(
    layer: dict[str, Any], alpha: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = layer.get("foreground_occlusion_mask")
    if not raw:
        return np.zeros(alpha.shape, dtype=bool), {
            "applied": False,
            "pixels": 0,
            "renderer_occlusion_regression": False,
        }
    receipt = layer.get("foreground_occlusion_receipt") or {}
    decisions = receipt.get("official_instance_decisions") or []
    valid_official = [
        row
        for row in decisions
        if row.get("is_distinct_from_target") is True
        and row.get("official_annotation_bound") is True
        and row.get("strictly_nearer_than_target") is True
        and row.get("accepted", True) is True
    ]
    valid_static = []
    for row in receipt.get("static_region_decisions") or []:
        threshold = int(
            row.get("minimum_verified_seed_pixels_per_component")
            or receipt.get("minimum_static_seed_pixels")
            or 1
        )
        accepted = [
            int(component.get("component_index") or 0)
            for component in row.get("components") or []
            if int(component.get("verified_nearer_seed_pixels") or 0) >= threshold
            and component.get("accepted") is True
        ]
        if (
            row.get("evidence_kind") == "unannotated_static_depth_seed_region"
            and row.get("accepted") is True
            and accepted
            and accepted == list(row.get("accepted_component_indices") or [])
            and int(row.get("selected_pixels") or 0) > 0
        ):
            valid_static.append(row)
    if receipt.get("schema_version") is not None:
        unsigned = dict(receipt)
        claimed = str(unsigned.pop("decision_sha256", ""))
        if (
            receipt.get("schema_version") != 1
            or receipt.get("policy")
            != "distinct_nearer_official_or_static_depth_v1"
            or claimed != canonical_sha256(unsigned)
            or receipt.get("quality_gate_pass") is not True
        ):
            raise ValueError("foreground receipt binding is invalid")
        static_hashes = (
            receipt.get("scene_depth_sha256"),
            receipt.get("asset_depth_sha256"),
            receipt.get("nearer_depth_support_sha256"),
        )
        if valid_static and any(len(str(value or "")) != 64 for value in static_hashes):
            raise ValueError("static foreground depth evidence is not hash-bound")
    if (
        (not valid_official and not valid_static)
        or receipt.get("independent_foreground_verified") is not True
    ):
        raise ValueError("foreground mask has no independent distinct-nearer proof")
    mask = _mask(Path(str(raw)).resolve(strict=True)) >= 0.5
    selected = mask & (alpha > 0.02)
    full_pixels = int((alpha > 0.02).sum())
    material = max(16, math.ceil(0.04 * full_pixels))
    pixels = int(selected.sum())
    if 0 < pixels < material:
        selected[:] = False
        pixels = 0
    # Target-instance support is diagnostic only. It cannot hide a distinct,
    # independently bound nearer official instance.
    exact_overlap = 0
    if layer.get("target_exact_mask"):
        exact = _mask(Path(str(layer["target_exact_mask"])).resolve(strict=True)) > 0.02
        exact_overlap = int((selected & exact).sum())
    if receipt.get("schema_version") is not None:
        claimed_pixels = int(receipt.get("selected_occlusion_pixels") or 0)
        if claimed_pixels != pixels:
            raise ValueError("foreground mask pixels differ from its receipt")
        if receipt.get("full_asset_support_sha256") != _array_sha256(
            (alpha > 0.02).astype(np.uint8)
        ) or receipt.get("selected_occlusion_mask_sha256") != _array_sha256(
            selected.astype(np.uint8)
        ):
            raise ValueError("foreground mask support differs from its receipt")
        if layer.get("target_exact_mask") and receipt.get(
            "target_exact_support_sha256"
        ) != _array_sha256(exact.astype(np.uint8)):
            raise ValueError("target exact support differs from its receipt")
    renderer_pixels = int(receipt.get("renderer_verified_occlusion_pixels") or 0)
    restored_pixels = int(receipt.get("restored_asset_pixels") or 0)
    regression = bool(
        renderer_pixels >= material
        and restored_pixels >= material
        and renderer_pixels - pixels >= material
    )
    return selected, {
        "applied": pixels >= material,
        "pixels": pixels,
        "material_pixels": material,
        "target_exact_overlap_kept_occluded": exact_overlap,
        "renderer_occlusion_regression": regression,
    }


def compose_results(results_path: Path, output_root: Path) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ids: set[str] = set()
    for result in iter_jsonl(results_path.resolve(strict=True)):
        sample_id = str(result.get("sample_id") or "")
        try:
            if not sample_id or sample_id in ids or result.get("status") != "complete":
                raise ValueError("invalid or duplicate render result")
            ids.add(sample_id)
            if result.get("render_domain") != "storm":
                raise ValueError("pair result is not from the STORM render domain")
            exposure = result.get("camera_exposure") or {}
            timestamp = int(exposure.get("sample_data_timestamp_us") or 0)
            normalized_time = float(
                exposure.get("storm_normalized_time_seconds") or math.nan
            )
            if (
                timestamp <= 0
                or int(exposure.get("camera_exposure_timestamp_us") or 0) != timestamp
                or not math.isfinite(normalized_time)
            ):
                raise ValueError("camera exposure and STORM time are not bound")
            cleanup = result.get("original_actor_cleanup_receipt") or {}
            unsigned_cleanup = dict(cleanup)
            claimed_cleanup = str(unsigned_cleanup.pop("decision_sha256", ""))
            if (
                claimed_cleanup != canonical_sha256(unsigned_cleanup)
                or cleanup.get("observation_technical_usable") is not True
                or cleanup.get("quality_gate_pass") is not True
            ):
                raise ValueError("original actor cleanup receipt is invalid")
            target_quality = (result.get("quality") or {}).get(
                "target_storm_vs_sensor_before_insertion"
            ) or {}
            if target_quality.get("quality_gate_pass") is not True:
                raise ValueError("STORM target quality gate did not pass")
            paths = result.get("paths") or {}
            gt_path = Path(str(paths.get("real_gt") or "")).resolve(strict=True)
            target_path = Path(str(paths.get("storm_baseline") or "")).resolve(
                strict=True
            )
            removed_path = Path(str(paths.get("actor_removed") or "")).resolve(
                strict=True
            )
            removed_mask_path = Path(
                str(paths.get("actor_removed_mask") or "")
            ).resolve(strict=True)
            expected_hashes = result.get("source_sha256") or {}
            for name, path in (
                ("real_gt", gt_path),
                ("storm_baseline", target_path),
                ("actor_removed", removed_path),
                ("actor_removed_mask", removed_mask_path),
            ):
                if (
                    len(str(expected_hashes.get(name) or "")) != 64
                    or sha256_file(path) != expected_hashes[name]
                ):
                    raise ValueError(f"source hash mismatch: {name}")
            gt = _rgb(gt_path)
            target = _rgb(target_path)
            inserted = _rgb(removed_path)
            removed_edit = _mask(removed_mask_path) > 0
            removal_delta = np.max(np.abs(inserted - target), axis=-1) > 0
            if int(removal_delta.sum()) < 20 or bool(
                (removal_delta & ~removed_edit).any()
            ):
                raise ValueError(
                    "actor removal changed unauthorized pixels or is ineffective"
                )
            layers: list[dict[str, Any]] = []
            for source_layer in result.get("asset_layers") or []:
                quality = source_layer.get("quality") or {}
                if (
                    source_layer.get("exact_identity_verified") is not True
                    or len(str(source_layer.get("asset_sha256") or "")) != 64
                    or source_layer.get("forward_axis") != "+X"
                    or source_layer.get("official_dimensions_applied") is not True
                    or source_layer.get("canonical_orientation_applied") is not True
                    or source_layer.get("bottom_ground_locked") is not True
                    or quality.get("degenerate_rectangle_mask") is True
                    or not geometry_pass(quality)
                ):
                    raise ValueError("identity or geometry candidate")
                dimensions = [
                    float(value)
                    for value in source_layer.get("official_dimensions_m") or []
                ]
                if len(dimensions) != 3 or any(value <= 0 for value in dimensions):
                    raise ValueError("official dimensions are invalid")
                rgb_path = Path(str(source_layer["rgb"])).resolve(strict=True)
                alpha_path = Path(str(source_layer["alpha"])).resolve(strict=True)
                layer_hashes = source_layer.get("content_sha256") or {}
                for name, path in (("rgb", rgb_path), ("alpha", alpha_path)):
                    if (
                        len(str(layer_hashes.get(name) or "")) != 64
                        or sha256_file(path) != layer_hashes[name]
                    ):
                        raise ValueError(f"asset layer hash mismatch: {name}")
                for name in ("foreground_occlusion_mask", "target_exact_mask"):
                    if source_layer.get(name):
                        path = Path(str(source_layer[name])).resolve(strict=True)
                        if (
                            len(str(layer_hashes.get(name) or "")) != 64
                            or sha256_file(path) != layer_hashes[name]
                        ):
                            raise ValueError(f"asset layer hash mismatch: {name}")
                alpha = _mask(alpha_path)
                rgb = _rgb(rgb_path)
                foreground, occlusion = _verified_foreground(source_layer, alpha)
                if occlusion["renderer_occlusion_regression"]:
                    raise ValueError("foreground occlusion regression candidate")
                alpha = alpha.copy()
                rgb = rgb.copy()
                alpha[foreground] = 0.0
                rgb[foreground] = 0.0
                complete = int(
                    quality.get("complete_projected_asset_pixels")
                    or (alpha > 0.02).sum()
                )
                visible = int((alpha > 0.02).sum())
                if visible < 20 or visible / max(complete, 1) < 0.15:
                    raise ValueError("asset is not materially visible")
                layers.append(
                    {
                        "obj_id": str(source_layer.get("obj_id") or ""),
                        "instance_token": str(source_layer.get("instance_token") or ""),
                        "asset_sha256": str(source_layer.get("asset_sha256") or ""),
                        "official_dimensions_m": dimensions,
                        "forward_axis": "+X",
                        "median_depth_m": float(
                            source_layer.get("median_depth_m") or math.inf
                        ),
                        "alpha": alpha,
                        "rgb": rgb,
                        "quality": quality,
                        "occlusion": occlusion,
                    }
                )
            if not layers:
                raise ValueError("render result has no accepted asset layers")
            visible_masks = [row["alpha"] > 0.02 for row in layers]
            if not edit_overlap_pass(visible_masks):
                raise ValueError("asset edit overlap exceeds train policy")
            removed = inserted.copy()
            for layer in sorted(
                layers, key=lambda row: row["median_depth_m"], reverse=True
            ):
                inserted = layer["rgb"] + inserted * (1.0 - layer["alpha"][..., None])
            insertion_delta = np.max(np.abs(inserted - removed), axis=-1) > 0
            # The 0.02 threshold is a visibility/geometry definition, not an
            # edit-authorisation boundary. Preserve the low-alpha Gaussian rim.
            union_insert = np.logical_or.reduce([row["alpha"] > 0 for row in layers])
            if int(insertion_delta.sum()) < 20:
                raise ValueError("asset insertion is ineffective")
            if bool(
                (
                    (np.max(np.abs(inserted - target), axis=-1) > 0)
                    & ~(removed_edit | union_insert)
                ).any()
            ):
                raise ValueError(
                    "composition changed pixels outside authorized support"
                )
            destinations = {
                role: output_root / role / f"{sample_id}.png"
                for role in ("gt", "input", "target")
            }
            _save_rgb(destinations["gt"], gt)
            _save_rgb(destinations["input"], inserted)
            _save_rgb(destinations["target"], target)
            record = {
                "schema_version": 1,
                "sample_id": sample_id,
                "scene_name": result.get("scene_name"),
                "frame_index": result.get("frame_index"),
                "camera_id": str(result.get("camera_id")),
                "selected_obj_ids": [row["obj_id"] for row in layers],
                "asset_layers": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"alpha", "rgb"}
                    }
                    for row in layers
                ],
                "paths": {
                    role: str(path.resolve()) for role, path in destinations.items()
                },
                "content_sha256": {
                    role: sha256_file(path) for role, path in destinations.items()
                },
                "render_lineage": {
                    "job_sha256": result.get("job_sha256"),
                    "checkpoint_sha256": result.get("checkpoint_sha256"),
                },
            }
            record["record_sha256"] = canonical_sha256(record)
            accepted.append(record)
        except Exception as exception:
            rejected.append(
                {
                    "sample_id": sample_id,
                    "reason": f"{type(exception).__name__}: {exception}",
                }
            )
    records_path = output_root / "records.jsonl"
    accepted_names = {f"{row['sample_id']}.png" for row in accepted}
    for role in ("gt", "input", "target"):
        role_root = output_root / role
        if role_root.is_dir():
            for path in role_root.glob("*.png"):
                if path.name not in accepted_names:
                    path.unlink()
    atomic_jsonl(records_path, accepted)
    atomic_jsonl(output_root / "candidates.jsonl", rejected)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "accepted_count": len(accepted),
        "candidate_count": len(rejected),
        "candidate_policy": "exclude",
        "records": str(records_path.resolve()),
        "records_sha256": canonical_sha256(accepted),
    }
    atomic_json(output_root / "summary.json", summary)
    return summary
