"""Atomic six-camera ring publication from already audited pair records."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

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


def _signed_row(row: dict[str, Any], field: str) -> bool:
    unsigned = dict(row)
    claimed = str(unsigned.pop(field, ""))
    return len(claimed) == 64 and claimed == canonical_sha256(unsigned)


def _strict_selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    path = path.resolve(strict=True)
    values: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = line.strip()
        group_id = str(value.get("group_id") or "") if isinstance(value, dict) else str(value)
        if not group_id:
            raise ValueError(f"empty group selection at {path}:{line_number}")
        values.append(group_id)
    if not values or len(values) != len(set(values)):
        raise ValueError("strict group selection is empty or duplicated")
    return set(values)


def _strict_indices(
    index_roots: Sequence[Path],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, int, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    groups: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    baseline_jobs: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    visible_jobs: dict[str, dict[str, Any]] = {}
    observed_splits: set[str] = set()
    for raw_root in index_roots:
        root = raw_root.resolve(strict=True)
        summary = read_json((root / "summary.json").resolve(strict=True))
        split = str(summary.get("split") or "")
        if (
            summary.get("status") != "complete"
            or split not in {"train", "val"}
            or split in observed_splits
            or (summary.get("content_verification") or {}).get("performed")
            is not True
        ):
            raise ValueError(f"strict index summary is invalid or duplicated: {root}")
        observed_splits.add(split)
        for name in (
            "source_index",
            "groups",
            "baseline_jobs",
            "visible_backfill_jobs",
        ):
            declared = (summary.get("outputs") or {}).get(name) or {}
            path = (root / f"{name}.jsonl").resolve(strict=True)
            if (
                Path(str(declared.get("path") or "")).resolve(strict=True) != path
                or sha256_file(path) != declared.get("sha256")
            ):
                raise ValueError(f"strict index output binding differs: {root}/{name}")
        for source in iter_jsonl(root / "source_index.jsonl"):
            sample_id = str(source.get("sample_id") or "")
            if (
                not sample_id
                or sample_id in sources
                or source.get("split") != split
                or not _signed_row(source, "record_sha256")
            ):
                raise ValueError(f"invalid strict source row: {sample_id}")
            sources[sample_id] = source
        for group in iter_jsonl(root / "groups.jsonl"):
            if group.get("split") != split or not _signed_row(group, "record_sha256"):
                raise ValueError(f"invalid strict group row: {group.get('group_id')}")
            groups.append(group)
        for job in iter_jsonl(root / "baseline_jobs.jsonl"):
            if job.get("split") != split or not _signed_row(job, "job_sha256"):
                raise ValueError(f"invalid strict baseline job: {job.get('job_id')}")
            for view in job.get("requested_views") or []:
                key = (
                    split,
                    str(job["context_id"]),
                    int(view["frame_index"]),
                    str(view["camera_id"]),
                )
                if key in baseline_jobs:
                    raise ValueError(f"baseline view is assigned to two jobs: {key}")
                baseline_jobs[key] = job
        for job in iter_jsonl(root / "visible_backfill_jobs.jsonl"):
            sample_id = str(job.get("sample_id") or "")
            if (
                not sample_id
                or sample_id in visible_jobs
                or job.get("split") != split
                or not _signed_row(job, "job_sha256")
            ):
                raise ValueError(f"invalid strict visible job: {sample_id}")
            visible_jobs[sample_id] = job
    group_ids = [str(row.get("group_id") or "") for row in groups]
    if not groups or any(not value for value in group_ids) or len(group_ids) != len(set(group_ids)):
        raise ValueError("strict group indices are empty or contain duplicate group IDs")
    return groups, sources, baseline_jobs, visible_jobs


def _baseline_results(
    paths: Sequence[Path],
    expected_jobs: dict[tuple[str, str, int, str], dict[str, Any]],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    jobs_by_id: dict[str, dict[str, Any]] = {}
    for job in expected_jobs.values():
        job_id = str(job.get("job_id") or "")
        previous = jobs_by_id.setdefault(job_id, job)
        if previous != job:
            raise ValueError(f"baseline job identity conflicts: {job_id}")
    for raw_path in paths:
        path = raw_path.resolve(strict=True)
        for row in iter_jsonl(path):
            expected_job = jobs_by_id.get(str(row.get("job_id") or ""))
            if (
                expected_job is None
                or row.get("status") not in {"complete", "failed"}
                or row.get("job_kind") != "sixcam_baseline_context"
                or not _signed_row(row, "result_sha256")
                or row.get("job_sha256") != expected_job.get("job_sha256")
                or row.get("checkpoint_sha256")
                != expected_job.get("checkpoint_sha256")
                or row.get("context_id") != expected_job.get("context_id")
                or row.get("scene_name") != expected_job.get("scene_name")
                or row.get("split") != expected_job.get("split")
            ):
                raise ValueError(f"invalid strict baseline result: {row.get('job_id')}")
            if row.get("status") == "failed":
                failure = row.get("error") or {}
                if row.get("views") or not str(failure.get("type") or ""):
                    raise ValueError(
                        f"invalid strict baseline failure: {row.get('job_id')}"
                    )
                continue
            for view in row.get("views") or []:
                key = (
                    str(row.get("split") or ""),
                    str(row.get("context_id") or ""),
                    int(view.get("frame_index", -1)),
                    str(view.get("camera_id") or ""),
                )
                job = expected_jobs.get(key)
                if (
                    job is None
                    or row.get("job_id") != job.get("job_id")
                    or job != expected_job
                    or not _signed_row(view, "view_sha256")
                ):
                    raise ValueError(f"baseline result is not bound to its request: {key}")
                previous = result.setdefault(key, view)
                if previous != view:
                    raise ValueError(f"baseline result content conflicts: {key}")
    return result


def _visible_results(
    paths: Sequence[Path], expected_jobs: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = raw_path.resolve(strict=True)
        for row in iter_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            job = expected_jobs.get(sample_id)
            lineage = row.get("render_lineage") or {}
            if (
                job is None
                or not _signed_row(row, "record_sha256")
                or lineage.get("job_sha256") != job.get("job_sha256")
                or lineage.get("checkpoint_sha256") != job.get("checkpoint_sha256")
                or str(row.get("scene_name") or "") != job.get("scene_name")
                or int(row.get("frame_index", -1)) != int(job.get("frame_index", -2))
                or str(row.get("camera_id") or "") != str(job.get("camera_id") or "")
                or list(row.get("selected_obj_ids") or [])
                != list(job.get("selected_obj_ids") or [])
            ):
                raise ValueError(f"visible result is not bound to its request: {sample_id}")
            layers = row.get("asset_layers") or []
            expected_assets = {
                (str(asset["obj_id"]), str(asset["instance_token"]), str(asset["asset_sha256"]))
                for asset in job.get("assets") or []
            }
            observed_assets = {
                (str(asset["obj_id"]), str(asset["instance_token"]), str(asset["asset_sha256"]))
                for asset in layers
            }
            hashes = row.get("content_sha256") or {}
            if (
                observed_assets != expected_assets
                or set(row.get("paths") or {}) != set(ROLES)
                or set(hashes) != set(ROLES)
                or hashes.get("input") == hashes.get("target")
            ):
                raise ValueError(f"visible result identity/content is invalid: {sample_id}")
            previous = result.setdefault(sample_id, row)
            if previous != row:
                raise ValueError(f"visible result conflicts: {sample_id}")
    return result


def _role_source(
    path: str, expected: str, source_kind: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    if len(expected) != 64:
        raise ValueError("resolved role source has no SHA-256")
    return {
        "path": path,
        "content_sha256": expected,
        "source_kind": source_kind,
        "evidence": evidence,
    }


def _accepted_source_has_complete_lineage(row: dict[str, Any]) -> bool:
    production_metadata = row.get("production_metadata") or {}
    production_record = str(row.get("production_record") or "")
    return (
        row.get("lineage_status") == "complete"
        and row.get("quality_gate_pass") is True
        and bool(production_record)
        and Path(production_record).is_file()
        and all(
            production_metadata.get(key)
            for key in ("manifest", "manifest_sha256", "row_sha256", "line_number")
        )
    )


def _resolve_strict_group(
    group: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    baseline_jobs: dict[tuple[str, str, int, str], dict[str, Any]],
    baseline_results: dict[tuple[str, str, int, str], dict[str, Any]],
    visible_jobs: dict[str, dict[str, Any]],
    visible_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identity = group.get("group_identity") or {}
    bound_asset_union = [
        {
            "global_uid": row["global_uid"],
            "obj_id": row["obj_id"],
            "instance_token": row["instance_token"],
            "canonical_asset_sha256": row["canonical_asset_sha256"],
        }
        for row in group.get("asset_union") or []
    ]
    if (
        not _signed_row(group, "record_sha256")
        or group.get("group_identity_sha256") != canonical_sha256(identity)
        or identity.get("split") != group.get("split")
        or identity.get("context_id") != group.get("context_id")
        or identity.get("scene_name") != group.get("scene_name")
        or identity.get("sample_token") != group.get("sample_token")
        or int(identity.get("frame_index", -1))
        != int(group.get("frame_index", -2))
        or identity.get("asset_union") != bound_asset_union
    ):
        raise ValueError("strict group identity binding differs")
    expected_group_id = (
        f"{group['split']}__{group['scene_name']}__f{int(group['frame_index']):06d}__"
        f"g-{group['group_identity_sha256'][:16]}"
    )
    if group.get("group_id") != expected_group_id:
        raise ValueError("strict group ID differs from its identity digest")
    views = group.get("views") or []
    if [str(row.get("camera_id")) for row in views] != [row[0] for row in CAMERA_RING]:
        raise ValueError("strict group camera order differs")
    union = {str(row["global_uid"]) for row in group.get("asset_union") or []}
    resolved_views: list[dict[str, Any]] = []
    for view in views:
        camera = str(view["camera_id"])
        camera_name = str(view["camera_name"])
        visible = tuple(map(str, view.get("camera_visible_asset_ids") or []))
        invisible = tuple(map(str, view.get("camera_invisible_asset_ids") or []))
        if (
            camera_name != CAMERA_NAMES[camera]
            or set(visible).isdisjoint(set(invisible)) is False
            or set(visible) | set(invisible) != union
            or str((view.get("exposure") or {}).get("sample_token") or "")
            != group.get("sample_token")
        ):
            raise ValueError(f"strict view identity/visibility differs: c{camera}")
        key = (
            str(group["split"]),
            str(group["context_id"]),
            int(group["frame_index"]),
            camera,
        )
        baseline_id = view.get("baseline_source_sample_id")
        baseline = sources.get(str(baseline_id)) if baseline_id else None
        if baseline is not None:
            if (
                not _accepted_source_has_complete_lineage(baseline)
                or baseline.get("split") != group.get("split")
                or baseline.get("context_id") != group.get("context_id")
                or int(baseline.get("frame_index", -1)) != int(group.get("frame_index", -2))
                or str(baseline.get("camera_id") or "") != camera
                or baseline.get("exposure") != view.get("exposure")
                or baseline.get("checkpoint_sha256") != identity.get("checkpoint_sha256")
            ):
                raise ValueError(f"accepted baseline source binding differs: c{camera}")
            gt = _role_source(
                baseline["paths"]["gt"],
                baseline["content_sha256"]["gt"],
                "accepted_nusc_pair",
                {
                    "sample_id": baseline["sample_id"],
                    "combination_id": baseline.get("combination_id"),
                    "production_record": baseline.get("production_record"),
                    "production_metadata": baseline.get("production_metadata"),
                    "authority": baseline["authority"],
                },
            )
            target = _role_source(
                baseline["paths"]["target"],
                baseline["content_sha256"]["target"],
                "accepted_nusc_pair",
                {
                    "sample_id": baseline["sample_id"],
                    "combination_id": baseline.get("combination_id"),
                    "production_record": baseline.get("production_record"),
                    "production_metadata": baseline.get("production_metadata"),
                    "authority": baseline["authority"],
                },
            )
        else:
            baseline_result = baseline_results.get(key)
            expected_job = baseline_jobs.get(key)
            if (
                baseline_result is None
                or expected_job is None
                or (baseline_result.get("quality") or {}).get("quality_gate_pass") is not True
            ):
                raise ValueError(f"required baseline backfill is unavailable or failed: c{camera}")
            gt = _role_source(
                baseline_result["paths"]["gt"],
                baseline_result["content_sha256"]["gt"],
                "sixcam_baseline_context",
                {"job_id": expected_job["job_id"], "job_sha256": expected_job["job_sha256"]},
            )
            target = _role_source(
                baseline_result["paths"]["target"],
                baseline_result["content_sha256"]["target"],
                "sixcam_baseline_context",
                {"job_id": expected_job["job_id"], "job_sha256": expected_job["job_sha256"]},
            )
        pair_id = view.get("pair_source_sample_id")
        pair = sources.get(str(pair_id)) if pair_id else None
        if not visible:
            if view.get("no_op") is not True:
                raise ValueError(f"invisible camera is not marked no-op: c{camera}")
            input_role = dict(target)
            input_role["source_kind"] = "visibility_no_op"
            input_role["evidence"] = {
                "reason": view.get("no_op_reason"),
                "target_source": target["evidence"],
            }
        elif pair is not None:
            if (
                not _accepted_source_has_complete_lineage(pair)
                or tuple(pair.get("selected_asset_ids") or ()) != visible
                or pair.get("context_id") != group.get("context_id")
                or int(pair.get("frame_index", -1)) != int(group.get("frame_index", -2))
                or str(pair.get("camera_id") or "") != camera
                or pair.get("checkpoint_sha256") != identity.get("checkpoint_sha256")
            ):
                raise ValueError(f"accepted visible source binding differs: c{camera}")
            input_role = _role_source(
                pair["paths"]["input"],
                pair["content_sha256"]["input"],
                "accepted_nusc_pair",
                {
                    "sample_id": pair["sample_id"],
                    "combination_id": pair.get("combination_id"),
                    "production_record": pair.get("production_record"),
                    "authority": pair["authority"],
                    "production_metadata": pair.get("production_metadata"),
                    "lineage_status": pair.get("lineage_status"),
                },
            )
        else:
            expected = next(
                (
                    job
                    for job in visible_jobs.values()
                    if job.get("split") == group.get("split")
                    and job.get("context_id") == group.get("context_id")
                    and int(job.get("frame_index", -1)) == int(group.get("frame_index", -2))
                    and str(job.get("camera_id") or "") == camera
                    and tuple(job.get("selected_asset_ids") or ()) == visible
                ),
                None,
            )
            rendered = visible_results.get(str((expected or {}).get("sample_id") or ""))
            if expected is None or rendered is None:
                raise ValueError(f"required visible backfill is unavailable: c{camera}")
            if (
                rendered["content_sha256"]["gt"] != gt["content_sha256"]
                or rendered["content_sha256"]["target"] != target["content_sha256"]
            ):
                raise ValueError(f"visible backfill background differs: c{camera}")
            input_role = _role_source(
                rendered["paths"]["input"],
                rendered["content_sha256"]["input"],
                "sixcam_visible_backfill",
                {
                    "sample_id": rendered["sample_id"],
                    "job_sha256": expected["job_sha256"],
                    "asset_layers": rendered["asset_layers"],
                },
            )
        if visible and input_role["content_sha256"] == target["content_sha256"]:
            raise ValueError(f"visible camera has a zero content edit: c{camera}")
        resolved_views.append(
            {
                **{
                    key_name: view.get(key_name)
                    for key_name in (
                        "camera_id",
                        "camera_name",
                        "visible",
                        "no_op",
                        "no_op_reason",
                        "camera_visible_asset_ids",
                        "camera_invisible_asset_ids",
                        "asset_state_sha256",
                        "exposure",
                    )
                },
                "role_sources": {"gt": gt, "input": input_role, "target": target},
            }
        )
    record = {
        "schema_version": 2,
        "group_id": group["group_id"],
        "group_identity": identity,
        "group_identity_sha256": group["group_identity_sha256"],
        "split": group["split"],
        "context_id": group["context_id"],
        "scene_name": group["scene_name"],
        "sample_token": group["sample_token"],
        "frame_index": group["frame_index"],
        "asset_union": group["asset_union"],
        "asset_state_sha256": group["asset_state_sha256"],
        "views": resolved_views,
        "review": {
            "status": "accepted",
            "policy": "strict_complete_group_or_quarantine",
        },
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _inspect_strict_source(task: tuple[Path, str]) -> tuple[Path, list[str]]:
    path, expected = task
    errors: list[str] = []
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            return path, ["not_regular_file"]
        if sha256_file(path) != expected:
            errors.append("hash_mismatch")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB":
                errors.append("invalid_png_or_mode")
            if image.size != IMAGE_SIZE:
                errors.append("wrong_dimensions")
    except Exception as exception:
        errors.append(f"{type(exception).__name__}: {exception}")
    return path, errors


def _bind_strict_published_paths(records: Sequence[dict[str, Any]]) -> None:
    """Bind every role receipt to its portable path in the formal release."""
    for row in records:
        for view in row["views"]:
            for role in ROLES:
                filename = (
                    f"{row['group_id']}__{view['camera_name']}__{role}.png"
                )
                view["role_sources"][role]["published_relative_path"] = (
                    f"{row['split']}/{filename}"
                )
        unsigned = dict(row)
        unsigned.pop("record_sha256", None)
        row["record_sha256"] = canonical_sha256(unsigned)


def _strict_release_statistics(
    records: Sequence[dict[str, Any]],
    quarantined: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    category_occurrences: Counter[str] = Counter()
    category_assets: dict[str, set[str]] = defaultdict(set)
    no_op_reasons: Counter[str] = Counter()
    source_kinds: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    asset_manifests: set[str] = set()
    checkpoint_sets: set[tuple[tuple[str, str], ...]] = set()
    visible_views = 0
    no_op_views = 0
    multi_asset_groups = 0
    for row in records:
        if len(row.get("asset_union") or []) > 1:
            multi_asset_groups += 1
        checkpoint_sets.add(
            tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in (
                        (row.get("group_identity") or {}).get("checkpoint_sha256")
                        or {}
                    ).items()
                )
            )
        )
        for asset in row.get("asset_union") or []:
            category = str(asset.get("category") or "unknown")
            category_occurrences[category] += 1
            category_assets[category].add(str(asset.get("global_uid") or ""))
            manifest = str(asset.get("exact_asset_manifest_sha256") or "")
            if manifest:
                asset_manifests.add(manifest)
        for view in row.get("views") or []:
            if view.get("visible") is True:
                visible_views += 1
            else:
                no_op_views += 1
                no_op_reasons[str(view.get("no_op_reason") or "unspecified")] += 1
            for role_source in (view.get("role_sources") or {}).values():
                source_kinds[str(role_source.get("source_kind") or "unknown")] += 1
    for row in quarantined:
        quarantine_reasons[str(row.get("reason") or "unspecified")] += 1
    checkpoints = [dict(values) for values in sorted(checkpoint_sets)]
    return {
        "schema_version": 1,
        "status": "complete",
        "group_count": len(records),
        "train_group_count": sum(row["split"] == "train" for row in records),
        "val_group_count": sum(row["split"] == "val" for row in records),
        "logical_image_count": len(records) * 18,
        "visible_edited_view_count": visible_views,
        "invisible_no_op_view_count": no_op_views,
        "multi_asset_group_count": multi_asset_groups,
        "category_asset_occurrence_count": dict(sorted(category_occurrences.items())),
        "category_unique_asset_count": {
            key: len(values) for key, values in sorted(category_assets.items())
        },
        "role_source_kind_count": dict(sorted(source_kinds.items())),
        "no_op_reason_count": dict(sorted(no_op_reasons.items())),
        "checkpoint_sha256_sets": checkpoints,
        "asset_manifest_sha256": sorted(asset_manifests),
        "quarantined_group_count": len(quarantined),
        "quarantine_reason_count": dict(sorted(quarantine_reasons.items())),
    }


def _copy_provenance_file(
    source: Path,
    target: Path,
    kind: str,
    entries: list[dict[str, Any]],
    metadata_root: Path,
) -> None:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"provenance source is not a regular file: {source}")
    if target.exists():
        raise ValueError(f"duplicate provenance target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_sha256 = sha256_file(source)
    copied_sha256 = sha256_file(target)
    if copied_sha256 != source_sha256:
        raise RuntimeError(f"provenance copy hash mismatch: {target}")
    entries.append(
        {
            "kind": kind,
            "original_path": str(source),
            "published_relative_path": str(target.relative_to(metadata_root.parent)),
            "sha256": copied_sha256,
            "size_bytes": target.stat().st_size,
        }
    )


def _stage_strict_provenance(
    *,
    staging: Path,
    index_roots: Sequence[Path],
    baseline_result_paths: Sequence[Path],
    visible_result_paths: Sequence[Path],
    receipt_paths: Sequence[Path],
    report_paths: Sequence[Path],
) -> dict[str, Any]:
    metadata_root = staging / "metadata"
    source_index_root = metadata_root / "source_index"
    receipts_root = metadata_root / "receipts"
    reports_root = metadata_root / "reports"
    for path in (source_index_root, receipts_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for raw_root in index_roots:
        root = raw_root.resolve(strict=True)
        split = str(read_json((root / "summary.json").resolve(strict=True)).get("split"))
        if split not in {"train", "val"}:
            raise ValueError(f"cannot label strict source index: {root}")
        for source in sorted(root.iterdir()):
            if source.is_file() and source.suffix in {".json", ".jsonl"}:
                _copy_provenance_file(
                    source,
                    source_index_root / split / source.name,
                    "strict_source_index",
                    entries,
                    metadata_root,
                )
    used_receipt_names: set[str] = set()
    for kind, paths in (
        ("baseline_result", baseline_result_paths),
        ("visible_result", visible_result_paths),
    ):
        for ordinal, raw_path in enumerate(paths):
            source = raw_path.resolve(strict=True)
            label = source.parent.name.replace("-", "_")
            name = f"{label}_{source.name}"
            if name in used_receipt_names:
                name = f"{kind}_{ordinal:02d}_{source.name}"
            used_receipt_names.add(name)
            _copy_provenance_file(
                source,
                receipts_root / name,
                kind,
                entries,
                metadata_root,
            )
    for ordinal, raw_path in enumerate(receipt_paths):
        source = raw_path.resolve(strict=True)
        parent_label = source.parent.name.replace("-", "_")
        name = f"{parent_label}_{source.name}"
        if name in used_receipt_names:
            name = f"receipt_{ordinal:02d}_{name}"
        used_receipt_names.add(name)
        _copy_provenance_file(
            source,
            receipts_root / name,
            "production_receipt",
            entries,
            metadata_root,
        )
    used_report_names: set[str] = set()
    for ordinal, raw_path in enumerate(report_paths):
        source = raw_path.resolve(strict=True)
        name = source.name
        if name in used_report_names:
            name = f"report_{ordinal:02d}_{name}"
        used_report_names.add(name)
        _copy_provenance_file(
            source,
            reports_root / name,
            "report",
            entries,
            metadata_root,
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "entry_count": len(entries),
        "entries": entries,
    }
    atomic_json(receipts_root / "provenance_manifest.json", manifest)
    return manifest


def _strict_release_readme(
    destination: Path,
    statistics: dict[str, Any],
    audit: dict[str, Any] | None,
    production_command: str | None,
) -> str:
    camera_lines = "\n".join(
        f"{ordinal}. `{name}`" for ordinal, (_, name) in enumerate(CAMERA_RING, 1)
    )
    category_lines = "\n".join(
        f"- `{category}`: {count:,} group-asset occurrences; "
        f"{statistics['category_unique_asset_count'].get(category, 0):,} unique assets"
        for category, count in statistics["category_asset_occurrence_count"].items()
    )
    checkpoint_lines = "\n".join(
        f"- `{name}`: `{digest}`"
        for values in statistics["checkpoint_sha256_sets"]
        for name, digest in sorted(values.items())
    )
    asset_manifest_lines = "\n".join(
        f"- `{value}`" for value in statistics["asset_manifest_sha256"]
    )
    quarantine_lines = "\n".join(
        f"- {reason}: {count:,} groups"
        for reason, count in statistics["quarantine_reason_count"].items()
    )
    run_command = production_command or (
        "python -m driveharm.sixcam_production --train-index <train-index> "
        "--val-index <val-index> --train-profile <train-profile> "
        "--val-profile <val-profile> --baseline-python <python> "
        "--visible-python <python> --render-contract <contract.json> "
        "--work-root <work-root> --destination <dataset-root> "
        "--receipt-root <receipt-root> --gpus 0,1,2,3,4,5,6,7"
    )
    audit_status = str((audit or {}).get("status") or "pending")
    candidate_count = int((audit or {}).get("candidate_count") or 0)
    return f"""# DriveHarm strict synchronized six-camera release

## Definition and layout

This is a strict, synchronized nuScenes six-camera dataset. Every `group_id`
binds one official split, scene, target sample/frame, STORM context, global asset
union, instance/PLY identity and frozen STORM/CVAC/DCN checkpoints. It is not a
renaming of unrelated single-camera pairs.

Each group has exactly 18 flat RGB PNG files: six `gt`, six edited/no-op `input`,
and six STORM `target` images. Files are named
`{{group_id}}__{{camera_name}}__{{gt|input|target}}.png` under `train/` or `val/`.

The fixed ring order is:

{camera_lines}

## Counts

- Groups: {statistics['group_count']:,} (train {statistics['train_group_count']:,}, val {statistics['val_group_count']:,})
- Logical PNGs: {statistics['logical_image_count']:,}
- Visible edited camera views: {statistics['visible_edited_view_count']:,}
- Invisible byte-identical `input=target` no-op views: {statistics['invisible_no_op_view_count']:,}
- Multi-asset groups: {statistics['multi_asset_group_count']:,}

Category counts count every asset occurrence in a published group and also list
the number of distinct `global_uid` values:

{category_lines}

## Provenance

Source triplets come from the audited `nusc_pair/train` and `nusc_pair/val`
releases. Missing six-camera STORM backgrounds and visible edits were rendered
with the frozen pipeline. Every group receipt records scene, sample token, frame,
camera exposure, visible/invisible asset partition, combination/source record,
`obj_id`, `instance_token`, `global_uid`, PLY path/hash, role source path/hash,
published relative path, per-asset depth/occlusion evidence where applicable,
and the accepted review decision.

Checkpoint SHA-256:

{checkpoint_lines}

Exact asset-manifest SHA-256:

{asset_manifest_lines}

Self-contained lineage is under `metadata/source_index/` and
`metadata/receipts/`; reports are under `metadata/reports/`. The group receipts
are `metadata/train_groups.jsonl` and `metadata/val_groups.jsonl`.

## Review, quarantine and audit

Visible views must have a non-zero edit and an accepted audited pair or signed
backfill with exact identity, geometry, grounding, appearance and occlusion
evidence. Invisible views are generated only from the frozen official visibility
partition and must have byte-identical `input` and `target`. Any missing or
rejected view quarantines the whole group; partial groups are never published.

Quarantined groups: {statistics['quarantined_group_count']:,}. They remain listed
in `metadata/audit/quarantined_groups.jsonl` and were not retried merely to fill a
quota. Reasons:

{quarantine_lines or '- none'}

The staging audit and independent replay check signed group identity, exact six
cameras/three roles, all logical paths and PNGs, content hashes, no-op equality,
visible non-equality, metadata provenance hashes, and train/val scene isolation.
Latest independent status: `{audit_status}` with {candidate_count:,} candidates.

Re-run the independent audit with:

```bash
driveharm sixcam-strict-audit --dataset-root {destination} \\
  --output-root <new-audit-directory> --workers 32
```

## Production and resume

The production invocation was:

```bash
{run_command}
```

The renderer and controller are resume-safe: re-run the same command with the
same work root. Signed completed shards are validated and reused; failed or
incomplete shards alone are scheduled again. Publication is built in a sibling
staging directory, fully audited there, and activated with an atomic rename.
"""


def audit_strict_sixcam_release(
    dataset_root: Path,
    output_root: Path,
    workers: int = 32,
) -> dict[str, Any]:
    """Independently replay a strict train+val flat release contract."""
    dataset_root = dataset_root.resolve(strict=True)
    if workers < 1:
        raise ValueError("audit workers must be positive")
    all_records: list[dict[str, Any]] = []
    expected_names: dict[str, set[str]] = {"train": set(), "val": set()}
    logical: list[tuple[Path, str]] = []
    errors: list[dict[str, Any]] = []
    scenes: dict[str, set[str]] = {"train": set(), "val": set()}
    group_ids: set[str] = set()
    metadata_root = dataset_root / "metadata"
    for name in ("source_index", "receipts", "audit", "reports"):
        path = metadata_root / name
        if not path.is_dir():
            errors.append({"metadata": str(path), "errors": ["required_directory_missing"]})
    for split in ("train", "val"):
        source_index = metadata_root / "source_index" / split
        group_manifest = metadata_root / f"{split}_groups.jsonl"
        has_groups = group_manifest.is_file() and group_manifest.stat().st_size > 0
        required = {
            "summary.json",
            "source_index.jsonl",
            "groups.jsonl",
            "baseline_jobs.jsonl",
            "visible_backfill_jobs.jsonl",
        }
        observed = {
            path.name for path in source_index.iterdir() if path.is_file()
        } if source_index.is_dir() else set()
        if has_groups and not required <= observed:
            errors.append(
                {
                    "metadata": str(source_index),
                    "errors": ["source_index_incomplete"],
                    "missing": sorted(required - observed),
                }
            )
    provenance_manifest = metadata_root / "receipts/provenance_manifest.json"
    if provenance_manifest.is_file():
        try:
            provenance = read_json(provenance_manifest)
            entries = provenance.get("entries") or []
            if (
                provenance.get("status") != "complete"
                or int(provenance.get("entry_count", -1)) != len(entries)
            ):
                raise ValueError("invalid provenance manifest header")
            for entry in entries:
                path = dataset_root / str(entry.get("published_relative_path") or "")
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path) != entry.get("sha256")
                    or path.stat().st_size != int(entry.get("size_bytes", -1))
                ):
                    raise ValueError(f"provenance entry mismatch: {path}")
        except Exception as exception:
            errors.append(
                {
                    "metadata": str(provenance_manifest),
                    "errors": [f"provenance_manifest_invalid: {exception}"],
                }
            )
    else:
        errors.append(
            {
                "metadata": str(provenance_manifest),
                "errors": ["provenance_manifest_missing"],
            }
        )
    release_summary = metadata_root / "reports/release_summary.json"
    if not release_summary.is_file():
        errors.append(
            {"metadata": str(release_summary), "errors": ["release_summary_missing"]}
        )
    readme = dataset_root / "README.md"
    readme_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    required_readme_terms = (
        "18 flat RGB PNG files",
        "CAM_FRONT_RIGHT",
        "Visible edited camera views",
        "Checkpoint SHA-256",
        "Exact asset-manifest SHA-256",
        "sixcam-strict-audit",
        "resume-safe",
        "metadata/audit/quarantined_groups.jsonl",
    )
    if any(term not in readme_text for term in required_readme_terms):
        errors.append({"metadata": str(readme), "errors": ["readme_incomplete"]})
    for split in ("train", "val"):
        manifest = dataset_root / "metadata" / f"{split}_groups.jsonl"
        for row in iter_jsonl(manifest.resolve(strict=True)):
            group_id = str(row.get("group_id") or "")
            group_errors: list[str] = []
            if (
                not group_id
                or group_id in group_ids
                or row.get("split") != split
                or not _signed_row(row, "record_sha256")
            ):
                group_errors.append("invalid_or_duplicate_group_binding")
            if (row.get("review") or {}).get("status") != "accepted":
                group_errors.append("group_review_not_accepted")
            group_ids.add(group_id)
            scenes[split].add(str(row.get("scene_name") or ""))
            identity = row.get("group_identity") or {}
            bound_asset_union = [
                {
                    "global_uid": asset["global_uid"],
                    "obj_id": asset["obj_id"],
                    "instance_token": asset["instance_token"],
                    "canonical_asset_sha256": asset[
                        "canonical_asset_sha256"
                    ],
                }
                for asset in row.get("asset_union") or []
            ]
            expected_group_id = (
                f"{split}__{row.get('scene_name')}__"
                f"f{int(row.get('frame_index', -1)):06d}__"
                f"g-{str(row.get('group_identity_sha256') or '')[:16]}"
            )
            if (
                row.get("group_identity_sha256") != canonical_sha256(identity)
                or group_id != expected_group_id
                or identity.get("sample_token") != row.get("sample_token")
                or identity.get("split") != split
                or identity.get("context_id") != row.get("context_id")
                or identity.get("scene_name") != row.get("scene_name")
                or int(identity.get("frame_index", -1))
                != int(row.get("frame_index", -2))
                or identity.get("asset_union") != bound_asset_union
            ):
                group_errors.append("group_identity_mismatch")
            views = row.get("views") or []
            if [str(view.get("camera_id")) for view in views] != [value[0] for value in CAMERA_RING]:
                group_errors.append("camera_ring_incomplete")
            union = {str(asset["global_uid"]) for asset in row.get("asset_union") or []}
            for view in views:
                camera = str(view.get("camera_id") or "")
                if view.get("camera_name") != CAMERA_NAMES.get(camera):
                    group_errors.append("camera_name_mismatch")
                    continue
                visible_assets = set(map(str, view.get("camera_visible_asset_ids") or []))
                invisible_assets = set(map(str, view.get("camera_invisible_asset_ids") or []))
                if visible_assets & invisible_assets or visible_assets | invisible_assets != union:
                    group_errors.append("camera_asset_partition_mismatch")
                if str((view.get("exposure") or {}).get("sample_token") or "") != row.get("sample_token"):
                    group_errors.append("camera_sample_token_mismatch")
                roles = view.get("role_sources") or {}
                if set(roles) != set(ROLES):
                    group_errors.append("role_membership_incomplete")
                    continue
                hashes = {role: str(roles[role].get("content_sha256") or "") for role in ROLES}
                if any(len(value) != 64 for value in hashes.values()):
                    group_errors.append("role_hash_incomplete")
                if any(
                    not str(roles[role].get("source_kind") or "")
                    or not isinstance(roles[role].get("evidence"), dict)
                    or not roles[role]["evidence"]
                    for role in ROLES
                ):
                    group_errors.append("role_evidence_incomplete")
                for role in ROLES:
                    source = roles[role]
                    if source.get("source_kind") == "accepted_nusc_pair":
                        evidence = source.get("evidence") or {}
                        production_metadata = evidence.get("production_metadata") or {}
                        if (
                            not evidence.get("sample_id")
                            or not evidence.get("combination_id")
                            or not evidence.get("production_record")
                            or not evidence.get("authority")
                            or any(
                                not production_metadata.get(key)
                                for key in (
                                    "manifest",
                                    "manifest_sha256",
                                    "row_sha256",
                                    "line_number",
                                )
                            )
                        ):
                            group_errors.append("accepted_source_lineage_incomplete")
                if view.get("no_op") is True:
                    if view.get("visible") is not False or hashes["input"] != hashes["target"]:
                        group_errors.append("invalid_no_op")
                elif view.get("visible") is not True or hashes["input"] == hashes["target"]:
                    group_errors.append("invalid_visible_edit")
                for role in ROLES:
                    filename = f"{group_id}__{view['camera_name']}__{role}.png"
                    expected_relative_path = f"{split}/{filename}"
                    if roles[role].get("published_relative_path") != expected_relative_path:
                        group_errors.append("published_role_path_mismatch")
                    expected_names[split].add(filename)
                    logical.append((dataset_root / split / filename, hashes[role]))
            if len(views) != 6:
                group_errors.append("group_does_not_have_six_views")
            if group_errors:
                errors.append({"group_id": group_id, "errors": sorted(set(group_errors))})
            all_records.append(row)
        observed = {path.name for path in (dataset_root / split).iterdir() if path.is_file()}
        if observed != expected_names[split]:
            errors.append(
                {
                    "split": split,
                    "errors": ["flat_file_membership_mismatch"],
                    "missing_count": len(expected_names[split] - observed),
                    "extra_count": len(observed - expected_names[split]),
                }
            )
    overlap = scenes["train"] & scenes["val"]
    if overlap:
        errors.append({"errors": ["train_val_scene_leak"], "scenes": sorted(overlap)})

    unique: dict[tuple[int, int], tuple[Path, str]] = {}
    for path, expected in logical:
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError("not_regular_file")
            inode = (path.stat().st_dev, path.stat().st_ino)
            previous = unique.setdefault(inode, (path, expected))
            if previous[1] != expected:
                raise ValueError("one inode is claimed with conflicting hashes")
        except Exception as exception:
            errors.append({"file": str(path), "errors": [str(exception)]})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected = list(executor.map(_inspect_strict_source, unique.values()))
    for path, found in inspected:
        if found:
            errors.append({"file": str(path), "errors": found})
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_root / "candidates.jsonl", errors)
    summary = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "group_count": len(all_records),
        "train_group_count": sum(row.get("split") == "train" for row in all_records),
        "val_group_count": sum(row.get("split") == "val" for row in all_records),
        "logical_image_count": len(logical),
        "expected_logical_image_count": len(all_records) * 18,
        "unique_inode_count": len(unique),
        "all_logical_paths_checked": len(logical) == len(all_records) * 18,
        "train_val_scene_overlap_count": len(overlap),
        "candidate_count": len(errors),
        "dataset_root": str(dataset_root),
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def build_strict_sixcam_review_sheets(
    dataset_root: Path, output_root: Path
) -> dict[str, Any]:
    """Build one lossless-resolution 6x3 pilot review sheet per group."""

    dataset_root = dataset_root.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    cell_width, cell_height = IMAGE_SIZE
    label_height = 24
    for split in ("train", "val"):
        manifest = (dataset_root / "metadata" / f"{split}_groups.jsonl").resolve(
            strict=True
        )
        for group in iter_jsonl(manifest):
            group_id = str(group.get("group_id") or "")
            views = group.get("views") or []
            if [str(view.get("camera_id")) for view in views] != [
                camera for camera, _name in CAMERA_RING
            ]:
                raise ValueError(f"review group camera ring differs: {group_id}")
            canvas = Image.new(
                "RGB",
                (cell_width * len(ROLES), (cell_height + label_height) * 6),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for row_index, view in enumerate(views):
                camera_name = str(view["camera_name"])
                roles = view.get("role_sources") or {}
                for column, role in enumerate(ROLES):
                    filename = f"{group_id}__{camera_name}__{role}.png"
                    path = (dataset_root / split / filename).resolve(strict=True)
                    if sha256_file(path) != roles[role]["content_sha256"]:
                        raise ValueError(f"review source hash differs: {filename}")
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != IMAGE_SIZE:
                            raise ValueError(f"review source image differs: {filename}")
                        x = column * cell_width
                        y = row_index * (cell_height + label_height) + label_height
                        canvas.paste(image, (x, y))
                    draw.text(
                        (column * cell_width + 4, row_index * (cell_height + label_height) + 4),
                        f"{camera_name} | {role} | {'visible' if view['visible'] else 'no-op'}",
                        fill="black",
                    )
            sheet = output_root / split / f"{group_id}.png"
            sheet.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(sheet, format="PNG")
            records.append(
                {
                    "group_id": group_id,
                    "split": split,
                    "sheet": str(sheet.resolve()),
                    "sheet_sha256": sha256_file(sheet),
                    "image_count": 18,
                    "native_cells": True,
                }
            )
    records_path = output_root / "records.jsonl"
    atomic_jsonl(records_path, records)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "operation": "strict_sixcam_review_sheets",
        "group_count": len(records),
        "image_count": len(records) * 18,
        "records": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
    }
    summary["summary_payload_sha256"] = canonical_sha256(summary)
    atomic_json(output_root / "summary.json", summary)
    return summary


def publish_strict_sixcam_release(
    *,
    index_roots: Sequence[Path],
    destination: Path,
    receipt_root: Path,
    baseline_result_paths: Sequence[Path] = (),
    visible_result_paths: Sequence[Path] = (),
    receipt_paths: Sequence[Path] = (),
    report_paths: Sequence[Path] = (),
    production_command: str | None = None,
    selection_path: Path | None = None,
    materialize: str = "hardlink",
    replace: bool = False,
    workers: int = 32,
) -> dict[str, Any]:
    """Resolve, quarantine and atomically publish strict 18-image groups."""
    if materialize not in {"hardlink", "copy"} or workers < 1:
        raise ValueError("strict publication options are invalid")
    groups, sources, baseline_jobs, visible_jobs = _strict_indices(index_roots)
    baseline_results = _baseline_results(baseline_result_paths, baseline_jobs)
    visible_results = _visible_results(visible_result_paths, visible_jobs)
    selected_ids = _strict_selection(selection_path)
    available_ids = {str(group["group_id"]) for group in groups}
    if selected_ids is not None and not selected_ids <= available_ids:
        raise ValueError(
            f"selection contains unknown groups: {sorted(selected_ids - available_ids)[:3]}"
        )
    records: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group["group_id"])
        if selected_ids is not None and group_id not in selected_ids:
            continue
        try:
            records.append(
                _resolve_strict_group(
                    group,
                    sources,
                    baseline_jobs,
                    baseline_results,
                    visible_jobs,
                    visible_results,
                )
            )
        except Exception as exception:
            quarantined.append(
                {
                    "group_id": group_id,
                    "split": group.get("split"),
                    "reason": f"{type(exception).__name__}: {exception}",
                }
            )
    if not records:
        raise ValueError("strict publication resolved no complete groups")
    records.sort(key=lambda row: (row["split"], row["group_id"]))
    _bind_strict_published_paths(records)
    statistics = _strict_release_statistics(records, quarantined)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{os.getpid()}"
    if staging.exists():
        raise ValueError(f"strict staging directory already exists: {staging}")
    for split in ("train", "val"):
        (staging / split).mkdir(parents=True)
    audit_root = staging / "metadata/audit"
    audit_root.mkdir(parents=True)
    source_expectations: dict[Path, str] = {}
    tasks: list[tuple[Path, Path, str]] = []
    try:
        for row in records:
            split_root = staging / row["split"]
            for view in row["views"]:
                for role in ROLES:
                    source_row = view["role_sources"][role]
                    source = Path(str(source_row["path"]))
                    expected = str(source_row["content_sha256"])
                    previous = source_expectations.setdefault(source, expected)
                    if previous != expected:
                        raise ValueError(f"source is claimed with conflicting hashes: {source}")
                    target = split_root / f"{row['group_id']}__{view['camera_name']}__{role}.png"
                    tasks.append((source, target, expected))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            inspected = list(
                executor.map(_inspect_strict_source, source_expectations.items())
            )
        failures = [
            {"file": str(path), "errors": found}
            for path, found in inspected
            if found
        ]
        if failures:
            atomic_jsonl(audit_root / "source_failures.jsonl", failures)
            raise RuntimeError(f"strict source preflight failed for {len(failures)} files")

        def materialize_one(task: tuple[Path, Path, str]) -> None:
            source, target, expected = task
            if materialize == "hardlink":
                os.link(source, target)
                if not os.path.samefile(source, target):
                    raise RuntimeError(f"hardlink identity mismatch: {target}")
            else:
                shutil.copy2(source, target)
                if sha256_file(target) != expected:
                    raise RuntimeError(f"copied content hash mismatch: {target}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(materialize_one, tasks))
        for split in ("train", "val"):
            atomic_jsonl(
                staging / "metadata" / f"{split}_groups.jsonl",
                (row for row in records if row["split"] == split),
            )
        atomic_jsonl(staging / "metadata/audit/quarantined_groups.jsonl", quarantined)
        provenance = _stage_strict_provenance(
            staging=staging,
            index_roots=index_roots,
            baseline_result_paths=baseline_result_paths,
            visible_result_paths=visible_result_paths,
            receipt_paths=receipt_paths,
            report_paths=report_paths,
        )
        atomic_json(staging / "metadata/reports/release_summary.json", statistics)
        (staging / "README.md").write_text(
            _strict_release_readme(
                destination, statistics, None, production_command
            ),
            encoding="utf-8",
        )
        audited = audit_strict_sixcam_release(staging, audit_root, workers)
        if audited["status"] != "pass":
            raise RuntimeError("strict staging audit failed")
        release_report = {**statistics, "audit": audited}
        atomic_json(staging / "metadata/reports/release_summary.json", release_report)
        (staging / "README.md").write_text(
            _strict_release_readme(
                destination, statistics, audited, production_command
            ),
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
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    receipt_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "destination": str(destination),
        "group_count": len(records),
        "image_count": len(records) * 18,
        "train_group_count": sum(row["split"] == "train" for row in records),
        "val_group_count": sum(row["split"] == "val" for row in records),
        "quarantined_group_count": len(quarantined),
        "materialize": materialize,
        "unique_source_file_count": len(source_expectations),
        "selection": str(selection_path.resolve()) if selection_path else None,
        "previous_release": str(backup) if backup else None,
        "audit": read_json(destination / "metadata/audit/summary.json"),
        "statistics": statistics,
        "provenance_entry_count": provenance["entry_count"],
    }
    atomic_json(receipt_root / "publish.json", summary)
    atomic_jsonl(receipt_root / "quarantined_groups.jsonl", quarantined)
    return summary
