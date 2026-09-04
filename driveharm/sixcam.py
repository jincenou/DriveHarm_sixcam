"""Atomic six-camera ring publication from already audited pair records."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable

from PIL import Image

from .contracts import (
    IMAGE_SIZE,
    ROLES,
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    iter_jsonl,
    read_json,
    sha256_file,
)


# nuScenes ring order, deliberately independent of the renderer's numeric order.
CAMERA_RING = (
    ("0", "CAM_FRONT"),
    ("2", "CAM_FRONT_RIGHT"),
    ("4", "CAM_BACK_RIGHT"),
    ("5", "CAM_BACK"),
    ("3", "CAM_BACK_LEFT"),
    ("1", "CAM_FRONT_LEFT"),
)
CAMERA_NAMES = dict(CAMERA_RING)


def _asset_values(row: dict[str, Any]) -> tuple[str, ...]:
    layers = row.get("asset_layers")
    if isinstance(layers, list):
        values = [str(layer.get("obj_id") or "") for layer in layers]
    else:
        values = [
            str(
                asset.get("global_uid")
                or asset.get("obj_id")
                or asset.get("asset_id")
                or asset.get("instance_token")
                or ""
            )
            for asset in row.get("assets") or []
        ]
    if not values or any(not value for value in values):
        return ()
    return tuple(sorted(set(values)))


def _visibility(
    manifest: Path,
) -> tuple[
    dict[str, str], dict[tuple[str, str, int], set[str]], dict[str, str]
]:
    payload = read_json(manifest.resolve(strict=True))
    aliases: dict[str, str] = {}
    categories: dict[str, str] = {}
    result: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for job in payload.get("jobs") or []:
        binding = job.get("exact_asset_binding") or {}
        canonical = str(
            binding.get("global_uid")
            or job.get("global_uid")
            or binding.get("obj_id")
            or job.get("obj_id")
            or ""
        )
        scene = str(job.get("scene_name") or "")
        if not canonical or not scene:
            raise ValueError("visibility job has incomplete scene/asset identity")
        category = str(binding.get("category") or job.get("category") or "")
        if not category:
            raise ValueError(f"visibility job has no category: {canonical}")
        previous_category = categories.setdefault(canonical, category)
        if previous_category != category:
            raise ValueError(f"asset category changes across jobs: {canonical}")
        for value in (
            canonical,
            binding.get("global_uid"),
            job.get("global_uid"),
            binding.get("obj_id"),
            job.get("obj_id"),
            binding.get("instance_token"),
            job.get("instance_token"),
        ):
            if value:
                previous = aliases.setdefault(str(value), canonical)
                if previous != canonical:
                    raise ValueError(f"ambiguous asset alias: {value}")
        for raw, usable in (job.get("target_usable_by_frame_camera") or {}).items():
            if usable is not True:
                continue
            frame, camera = str(raw).split(":c", 1)
            if camera not in CAMERA_NAMES:
                raise ValueError(f"unknown camera in visibility manifest: {camera}")
            result[(scene, canonical, int(frame))].add(camera)
    if not result:
        raise ValueError("visibility manifest contains no usable camera evidence")
    return aliases, dict(result), categories


def _signature(row: dict[str, Any], aliases: dict[str, str]) -> tuple[str, ...]:
    values = _asset_values(row)
    if not values or any(value not in aliases for value in values):
        return ()
    return tuple(sorted(aliases[value] for value in values))


def _source_path(root: Path, row: dict[str, Any], role: str) -> Path:
    return root / role / f"{row['sample_id']}.png"


def _materialize(source: Path, target: Path, mode: str, expected: str) -> None:
    source_mode = source.lstat().st_mode
    if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode):
        raise ValueError(f"source is not a regular file: {source}")
    if sha256_file(source) != expected:
        raise ValueError(f"source hash differs from audited record: {source}")
    if mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)
    if sha256_file(target) != expected:
        raise RuntimeError(f"materialized hash mismatch: {target}")


def _candidate_groups(
    rows: list[dict[str, Any]],
    aliases: dict[str, str],
    visibility: dict[tuple[str, str, int], set[str]],
    scenes: set[str],
    categories: dict[str, str],
    selected_categories: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    baselines: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    exact: dict[
        tuple[str, int, str, tuple[str, ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    candidate_sets: set[tuple[str, int, tuple[str, ...]]] = set()
    for row in rows:
        scene = str(row.get("scene_name") or "")
        frame = int(row.get("frame_index", -1))
        camera = str(row.get("camera_id") or "")
        if not scene or frame < 0 or camera not in CAMERA_NAMES:
            raise ValueError(f"invalid accepted record identity: {row.get('sample_id')}")
        if scenes and scene not in scenes:
            continue
        baselines[(scene, frame, camera)].append(row)
        signature = _signature(row, aliases)
        if signature:
            exact[(scene, frame, camera, signature)].append(row)
            candidate_sets.add((scene, frame, signature))

    scene_frames = {(scene, frame) for scene, frame, _ in baselines}
    eligible_frames = {
        (scene, frame)
        for scene, frame in scene_frames
        if all(
            len(
                {
                    (row["content_sha256"]["gt"], row["content_sha256"]["target"])
                    for row in baselines.get((scene, frame, camera), [])
                }
            )
            == 1
            for camera in CAMERA_NAMES
        )
    }
    candidate_sets = {
        value
        for value in candidate_sets
        if value[:2] in eligible_frames
        and (
            not selected_categories
            or all(categories.get(asset) in selected_categories for asset in value[2])
        )
    }
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for scene, frame, selected in sorted(candidate_sets):
        reason = ""
        sources: dict[str, dict[str, Any]] = {}
        visible_by_camera = {
            camera: tuple(
                asset
                for asset in selected
                if camera in visibility.get((scene, asset, frame), set())
            )
            for camera in CAMERA_NAMES
        }
        if any(not visibility.get((scene, asset, frame)) for asset in selected):
            reason = "selected asset has no explicit visibility evidence"
        for camera in CAMERA_NAMES:
            donors = baselines.get((scene, frame, camera), [])
            background_signatures = {
                (row["content_sha256"]["gt"], row["content_sha256"]["target"])
                for row in donors
            }
            assert len(background_signatures) == 1
            visible = visible_by_camera[camera]
            pair_rows = exact.get((scene, frame, camera, visible), []) if visible else []
            if visible and not pair_rows:
                reason = f"visible camera {camera} has no matching audited pair"
                break
            if visible and len({row["content_sha256"]["input"] for row in pair_rows}) != 1:
                reason = f"visible camera {camera} has conflicting audited inputs"
                break
            baseline = min(donors, key=lambda row: str(row["sample_id"]))
            pair = min(pair_rows, key=lambda row: str(row["sample_id"])) if visible else None
            sources[camera] = {"baseline": baseline, "pair": pair}
        if reason:
            excluded.append(
                {
                    "scene_name": scene,
                    "frame_index": frame,
                    "selected_asset_ids": list(selected),
                    "reason": reason,
                }
            )
            continue
        group_id = f"{scene}__f{frame:03d}__g-{canonical_sha256(list(selected))[:12]}"
        accepted.append(
            {
                "group_id": group_id,
                "scene_name": scene,
                "frame_index": frame,
                "selected_asset_ids": list(selected),
                "visible_assets_by_camera": {
                    camera: list(values) for camera, values in visible_by_camera.items()
                },
                "sources": sources,
            }
        )
    return accepted, excluded, {
        "source_scene_frame_count": len(scene_frames),
        "complete_unambiguous_ring_frame_count": len(eligible_frames),
        "candidate_asset_set_count": len(candidate_sets),
    }


def build_sixcam_release(
    source_root: Path,
    records_path: Path,
    visibility_manifest: Path,
    destination: Path,
    receipt_root: Path,
    maximum_groups: int = 0,
    scenes: Iterable[str] = (),
    categories: Iterable[str] = (),
    materialize: str = "hardlink",
    replace: bool = False,
    workers: int = 32,
) -> dict[str, Any]:
    """Publish complete 18-image groups; incomplete groups are never emitted."""
    if materialize not in {"hardlink", "copy"} or maximum_groups < 0:
        raise ValueError("invalid publication options")
    source_root = source_root.resolve(strict=True)
    rows = list(iter_jsonl(records_path.resolve(strict=True)))
    if not rows:
        raise ValueError("accepted records are empty")
    aliases, visibility, asset_categories = _visibility(visibility_manifest)
    groups, excluded, capacity = _candidate_groups(
        rows,
        aliases,
        visibility,
        set(scenes),
        asset_categories,
        set(categories),
    )
    groups.sort(
        key=lambda group: (
            -sum(bool(value) for value in group["visible_assets_by_camera"].values()),
            group["group_id"],
        )
    )
    complete_group_count = len(groups)
    if maximum_groups:
        groups = groups[:maximum_groups]
    if not groups:
        raise ValueError("no complete six-camera groups can be published")

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{os.getpid()}"
    if staging.exists():
        raise ValueError(f"staging directory already exists: {staging}")
    staging.mkdir()
    records: list[dict[str, Any]] = []
    tasks: list[tuple[Path, Path, str, str]] = []
    try:
        for group in groups:
            views: list[dict[str, Any]] = []
            for camera, camera_name in CAMERA_RING:
                source = group["sources"][camera]
                baseline = source["baseline"]
                pair = source["pair"]
                role_sources = {
                    "gt": baseline,
                    "target": baseline,
                    "input": pair or baseline,
                }
                files: dict[str, str] = {}
                hashes: dict[str, str] = {}
                for role in ROLES:
                    source_row = role_sources[role]
                    source_role = "target" if role == "input" and pair is None else role
                    expected = str(source_row["content_sha256"][source_role])
                    filename = f"{group['group_id']}__{camera_name}__{role}.png"
                    files[role] = filename
                    hashes[role] = expected
                    tasks.append(
                        (
                            _source_path(source_root, source_row, source_role),
                            staging / filename,
                            expected,
                            materialize,
                        )
                    )
                views.append(
                    {
                        "camera_id": camera,
                        "camera_name": camera_name,
                        "asset_visible": pair is not None,
                        "applied_asset_ids": group["visible_assets_by_camera"][camera],
                        "source_pair_sample_id": pair["sample_id"] if pair else None,
                        "baseline_sample_id": baseline["sample_id"],
                        "files": files,
                        "content_sha256": hashes,
                    }
                )
            record = {
                "schema_version": 1,
                "group_id": group["group_id"],
                "scene_name": group["scene_name"],
                "frame_index": group["frame_index"],
                "selected_asset_ids": group["selected_asset_ids"],
                "views": views,
            }
            record["record_sha256"] = canonical_sha256(record)
            records.append(record)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            list(executor.map(lambda task: _materialize(task[0], task[1], task[3], task[2]), tasks))
        (staging / "README.md").write_text(
            "# DriveHarm six-camera flat release\n\n"
            f"Groups: {len(records):,}\n\n"
            "Each group contains exactly six nuScenes ring cameras and three roles "
            "(`gt`, `input`, `target`), for 18 flat PNG files. Invisible views use "
            "the unedited STORM target as input; assets are never forced into them.\n",
            encoding="utf-8",
        )
        backup: Path | None = None
        if destination.exists():
            if not replace:
                raise ValueError(f"destination exists: {destination}")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = destination.parent / f".{destination.name}.previous.{stamp}"
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    receipt_root.mkdir(parents=True, exist_ok=True)
    manifest = receipt_root / "groups.jsonl"
    atomic_jsonl(manifest, records)
    atomic_jsonl(receipt_root / "excluded_groups.jsonl", excluded)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "group_count": len(records),
        "image_count": len(records) * 18,
        "candidate_group_count": capacity["candidate_asset_set_count"],
        "complete_group_count_before_limit": complete_group_count,
        "excluded_group_count": len(excluded),
        "destination": str(destination),
        "groups": str(manifest.resolve()),
        "groups_sha256": canonical_sha256(records),
        "visibility_manifest": str(visibility_manifest.resolve()),
        "visibility_manifest_sha256": sha256_file(visibility_manifest),
        "source_records": str(records_path.resolve()),
        "source_records_sha256": sha256_file(records_path),
        "category_filter": sorted(set(categories)),
        "previous_release": str(backup) if backup else None,
        **capacity,
    }
    atomic_json(receipt_root / "build_summary.json", summary)
    return summary


def audit_sixcam_release(
    dataset_root: Path,
    groups_path: Path,
    source_records_path: Path,
    visibility_manifest: Path,
    output_root: Path,
    workers: int = 32,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    groups = list(iter_jsonl(groups_path.resolve(strict=True)))
    if not groups or len({row.get("group_id") for row in groups}) != len(groups):
        raise ValueError("group manifest is empty or duplicated")
    source_rows = list(iter_jsonl(source_records_path.resolve(strict=True)))
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in source_by_id:
            raise ValueError("source records contain an empty or duplicate sample_id")
        source_by_id[sample_id] = row
    aliases, visibility, _ = _visibility(visibility_manifest)
    expected_names: set[str] = set()
    tasks: list[tuple[Path, str]] = []
    errors: list[dict[str, Any]] = []
    expected_cameras = {camera for camera, _ in CAMERA_RING}
    for group in groups:
        unsigned = dict(group)
        claimed = str(unsigned.pop("record_sha256", ""))
        group_errors: list[str] = []
        if claimed != canonical_sha256(unsigned):
            group_errors.append("record_binding_mismatch")
        scene = str(group.get("scene_name") or "")
        frame = int(group.get("frame_index", -1))
        selected = tuple(group.get("selected_asset_ids") or ())
        if (
            not scene
            or frame < 0
            or not selected
            or tuple(sorted(set(selected))) != selected
            or any(asset not in aliases.values() for asset in selected)
        ):
            group_errors.append("invalid_group_identity")
        expected_group_id = (
            f"{scene}__f{frame:03d}__g-{canonical_sha256(list(selected))[:12]}"
        )
        if group.get("group_id") != expected_group_id:
            group_errors.append("group_id_mismatch")
        views = group.get("views") or []
        camera_order = [str(view.get("camera_id")) for view in views]
        if (
            len(views) != 6
            or set(camera_order) != expected_cameras
            or camera_order != [camera for camera, _ in CAMERA_RING]
        ):
            group_errors.append("camera_ring_incomplete")
        for view in views:
            camera = str(view.get("camera_id"))
            if view.get("camera_name") != CAMERA_NAMES.get(camera):
                group_errors.append("camera_name_mismatch")
            if camera not in CAMERA_NAMES:
                continue
            files = view.get("files") or {}
            hashes = view.get("content_sha256") or {}
            if set(files) != set(ROLES) or set(hashes) != set(ROLES):
                group_errors.append("role_membership_incomplete")
                continue
            expected = {
                role: f"{group['group_id']}__{CAMERA_NAMES[camera]}__{role}.png"
                for role in ROLES
            }
            if files != expected:
                group_errors.append("flat_filename_mismatch")
            expected_assets = tuple(
                asset
                for asset in selected
                if camera in visibility.get((scene, asset, frame), set())
            )
            applied_assets = tuple(view.get("applied_asset_ids") or ())
            if applied_assets != expected_assets:
                group_errors.append("visibility_binding_mismatch")
            baseline = source_by_id.get(str(view.get("baseline_sample_id") or ""))
            if baseline is None:
                group_errors.append("baseline_source_missing")
            elif (
                str(baseline.get("scene_name")) != scene
                or int(baseline.get("frame_index", -1)) != frame
                or str(baseline.get("camera_id")) != camera
                or hashes.get("gt") != baseline.get("content_sha256", {}).get("gt")
                or hashes.get("target")
                != baseline.get("content_sha256", {}).get("target")
            ):
                group_errors.append("baseline_source_binding_mismatch")
            if view.get("asset_visible") is True:
                pair = source_by_id.get(str(view.get("source_pair_sample_id") or ""))
                if not applied_assets or pair is None:
                    group_errors.append("visible_view_has_no_asset_pair")
                elif (
                    str(pair.get("scene_name")) != scene
                    or int(pair.get("frame_index", -1)) != frame
                    or str(pair.get("camera_id")) != camera
                    or _signature(pair, aliases) != applied_assets
                    or hashes.get("input") != pair.get("content_sha256", {}).get("input")
                    or hashes.get("gt") != pair.get("content_sha256", {}).get("gt")
                    or hashes.get("target")
                    != pair.get("content_sha256", {}).get("target")
                ):
                    group_errors.append("pair_source_binding_mismatch")
                if hashes.get("input") == hashes.get("target"):
                    group_errors.append("visible_view_has_no_effective_edit")
            elif (
                view.get("applied_asset_ids")
                or view.get("source_pair_sample_id") is not None
                or hashes.get("input") != hashes.get("target")
            ):
                group_errors.append("invisible_view_was_edited")
            for role in ROLES:
                if len(str(hashes.get(role) or "")) != 64:
                    group_errors.append("invalid_content_hash")
                    continue
                filename = files[role]
                expected_names.add(filename)
                tasks.append((dataset_root / filename, hashes[role]))
        if group_errors:
            errors.append({"group_id": group.get("group_id"), "errors": sorted(set(group_errors))})
    observed = {path.name for path in dataset_root.glob("*.png")}
    if observed != expected_names:
        errors.append(
            {
                "group_id": "__release__",
                "errors": ["flat_png_membership_mismatch"],
                "missing_count": len(expected_names - observed),
                "extra_count": len(observed - expected_names),
            }
        )

    def inspect(task: tuple[Path, str]) -> list[str]:
        path, expected = task
        found: list[str] = []
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                return ["not_regular_file"]
            if sha256_file(path) != expected:
                found.append("hash_mismatch")
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB":
                    found.append("invalid_png_or_mode")
                if image.size != IMAGE_SIZE:
                    found.append("wrong_dimensions")
        except Exception as exception:
            found.append(f"{type(exception).__name__}: {exception}")
        return found

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        image_results = list(executor.map(inspect, tasks))
    for (path, _), found in zip(tasks, image_results):
        if found:
            errors.append({"file": path.name, "errors": found})
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_root / "candidates.jsonl", errors)
    summary = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "group_count": len(groups),
        "expected_image_count": len(groups) * 18,
        "checked_image_count": len(tasks),
        "all_images_checked": len(tasks) == len(groups) * 18,
        "candidate_count": len(errors),
        "flat_directory": True,
        "source_record_count": len(source_rows),
        "source_records_sha256": sha256_file(source_records_path),
        "visibility_manifest_sha256": sha256_file(visibility_manifest),
    }
    atomic_json(output_root / "audit_summary.json", summary)
    return summary
