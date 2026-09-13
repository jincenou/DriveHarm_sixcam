"""Build a strict, render-aware six-camera production index.

The legacy six-camera publisher intentionally remains available for reproducing
the historical release.  This module is the production path: it retains the
STORM context, replays context-specific visibility, resolves each camera
exposure against the official nuScenes tables, and plans missing work at context
grain so that backgrounds are never rendered once per group.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Iterator, Sequence

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
from .sixcam import CAMERA_NAMES, CAMERA_RING


SAMPLE_ID = re.compile(
    r"^(?P<scene>scene-\d+)__"
    r"(?P<window>w\d{6})__"
    r"(?P<context_digest>[0-9a-f]{12})__"
    r"(?P<combination>cmb-[0-9a-f]{12})__"
    r"f(?P<frame>\d+)_c(?P<camera>[0-5])$"
)
CAMERA_BY_NAME = {name: camera for camera, name in CAMERA_RING}


def parse_sample_id(value: str) -> dict[str, Any]:
    match = SAMPLE_ID.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid production sample_id: {value}")
    result: dict[str, Any] = match.groupdict()
    result["frame_index"] = int(result.pop("frame"))
    result["camera_id"] = result.pop("camera")
    result["context_id"] = (
        f"{result['scene']}__{result['window']}__{result['context_digest']}"
    )
    result["combination_id"] = (
        f"{result['context_id']}__{result.pop('combination')}"
    )
    return result


def iter_json_array(path: Path, field: str | None = None) -> Iterator[dict[str, Any]]:
    """Stream objects from a root JSON array or a named top-level array.

    The frozen plans are larger than one GB.  Loading them with ``json.load``
    needlessly multiplies memory use, so this decoder retains at most one array
    member plus a small input buffer.
    """

    decoder = json.JSONDecoder()
    marker = f'"{field}"' if field else None
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        start = -1
        while start < 0:
            block = stream.read(1024 * 1024)
            if not block:
                raise ValueError(f"JSON array {field!r} not found: {path}")
            buffer += block
            if marker:
                key = buffer.find(marker)
                if key >= 0:
                    start = buffer.find("[", key + len(marker))
            else:
                start = buffer.find("[")
            if start < 0:
                buffer = buffer[-max(128, len(marker or "")) :]
        buffer = buffer[start + 1 :]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = stream.read(1024 * 1024)
                if not block:
                    raise ValueError(f"truncated JSON array in {path}")
                buffer += block
                continue
            if not isinstance(value, dict):
                raise ValueError(f"expected object in JSON array: {path}")
            yield value
            buffer = buffer[end:]


def _asset_values(row: dict[str, Any]) -> tuple[str, ...]:
    assets = row.get("assets") or row.get("selected_exact_assets") or []
    if assets:
        values = [
            asset.get("global_uid")
            or asset.get("obj_id")
            or asset.get("asset_id")
            or asset.get("instance_token")
            for asset in assets
        ]
    else:
        values = row.get("selected_obj_ids") or row.get("selected_asset_ids") or []
    normalized = tuple(sorted(set(str(value) for value in values if value)))
    return normalized


def _asset_binding(job: dict[str, Any], canonical: str) -> dict[str, Any]:
    source = job.get("exact_asset_binding") or {}
    canonical_metadata = source.get("canonical") or {}
    forward_axis = str(
        canonical_metadata.get("front_axis")
        or canonical_metadata.get("heading")
        or job.get("forward_axis")
        or "+X"
    )
    result = {
        "global_uid": canonical,
        "obj_id": str(source.get("obj_id") or job.get("obj_id") or ""),
        "instance_token": str(
            source.get("instance_token") or job.get("instance_token") or ""
        ),
        "category": str(source.get("category") or job.get("category") or ""),
        "canonical_asset_path": str(
            source.get("asset_path") or job.get("canonical_asset_ply") or ""
        ),
        "canonical_asset_sha256": str(
            source.get("asset_sha256") or job.get("canonical_asset_sha256") or ""
        ),
        "exact_asset_manifest_sha256": str(
            source.get("manifest_sha256")
            or job.get("exact_asset_manifest_sha256")
            or ""
        ),
        "forward_axis": forward_axis,
    }
    if (
        any(not result[key] for key in ("obj_id", "instance_token", "category"))
        or len(result["canonical_asset_sha256"]) != 64
        or len(result["exact_asset_manifest_sha256"]) != 64
        or forward_axis != "+X"
    ):
        raise ValueError(f"incomplete exact asset binding: {canonical}")
    return result


def load_context_visibility(
    manifest_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, int], dict[str, bool]],
    dict[str, dict[str, Any]],
]:
    manifest_path = manifest_path.resolve(strict=True)
    payload = read_json(manifest_path)
    aliases: dict[str, str] = {}
    assets: dict[str, dict[str, Any]] = {}
    visibility: dict[tuple[str, str, int], dict[str, bool]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for job in payload.get("jobs") or []:
        source = job.get("exact_asset_binding") or {}
        canonical = str(
            source.get("global_uid")
            or job.get("global_uid")
            or source.get("obj_id")
            or job.get("obj_id")
            or ""
        )
        context = str(job.get("multi_case_id") or "")
        scene = str(job.get("scene_name") or "")
        if not canonical or not context or not scene or not context.startswith(scene + "__"):
            raise ValueError("visibility job has incomplete context/asset identity")
        binding = _asset_binding(job, canonical)
        previous_binding = assets.setdefault(canonical, binding)
        if previous_binding != binding:
            raise ValueError(f"asset binding changes across contexts: {canonical}")
        for value in (
            canonical,
            source.get("global_uid"),
            job.get("global_uid"),
            source.get("obj_id"),
            job.get("obj_id"),
            source.get("instance_token"),
            job.get("instance_token"),
        ):
            if value:
                previous = aliases.setdefault(str(value), canonical)
                if previous != canonical:
                    raise ValueError(f"ambiguous asset alias: {value}")
        context_binding = {
            "context_id": context,
            "scene_name": scene,
            "multi_case_index": int(job.get("multi_case_index", -1)),
        }
        previous_context = contexts.setdefault(context, context_binding)
        if previous_context != context_binding:
            raise ValueError(f"context binding changes: {context}")
        for raw, usable in (job.get("target_usable_by_frame_camera") or {}).items():
            frame_value, camera = str(raw).split(":c", 1)
            if camera not in CAMERA_NAMES:
                raise ValueError(f"unknown camera in visibility manifest: {camera}")
            key = (context, canonical, int(frame_value))
            values = visibility.setdefault(key, {})
            if camera in values and values[camera] is not bool(usable):
                raise ValueError(f"visibility decision changes: {key}:c{camera}")
            values[camera] = bool(usable)
    if not visibility:
        raise ValueError("visibility manifest contains no frame/camera evidence")
    return aliases, assets, visibility, contexts


def _checkpoint_contract(
    render_contract_path: Path, verify_content: bool
) -> tuple[dict[str, str], dict[str, str]]:
    path = render_contract_path.resolve(strict=True)
    artifacts = read_json(path).get("artifacts") or {}
    hashes: dict[str, str] = {}
    paths: dict[str, str] = {}
    for name in ("storm", "cvac", "dcn"):
        artifact = artifacts.get(name) or {}
        artifact_path = Path(str(artifact.get("path") or "")).resolve(strict=True)
        expected = str(artifact.get("sha256") or "")
        if len(expected) != 64:
            raise ValueError(f"invalid checkpoint hash: {name}")
        if verify_content and sha256_file(artifact_path) != expected:
            raise ValueError(f"checkpoint content mismatch: {name}")
        hashes[name] = expected
        paths[name] = str(artifact_path)
    return hashes, paths


def _official_scenes(path: Path, split: str) -> set[str]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get(split)
    if not isinstance(value, list) or not value:
        raise ValueError(f"official scene manifest has no {split!r} list")
    scenes = {str(scene) for scene in value}
    if "" in scenes or len(scenes) != len(value):
        raise ValueError("official scene manifest is empty or duplicated")
    return scenes


def _release_rows(
    records_path: Path,
    split: str,
    scenes: set[str],
    aliases: dict[str, str],
    assets: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records_path = records_path.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(iter_jsonl(records_path), 1):
        sample_id = str(row.get("sample_id") or "")
        try:
            if not sample_id or sample_id in seen:
                raise ValueError("empty or duplicate sample_id")
            seen.add(sample_id)
            parsed = parse_sample_id(sample_id)
            scene = parsed["scene"]
            if scene not in scenes or (row.get("split") not in (None, split)):
                raise ValueError("record is outside the official split")
            for key, expected in (
                ("scene_name", scene),
                ("frame_index", parsed["frame_index"]),
                ("camera_id", parsed["camera_id"]),
            ):
                if row.get(key) is not None and str(row[key]) != str(expected):
                    raise ValueError(f"top-level {key} differs from sample_id")
            raw_assets = _asset_values(row)
            if not raw_assets or any(value not in aliases for value in raw_assets):
                raise ValueError("asset signature is absent from frozen visibility")
            selected = tuple(sorted(aliases[value] for value in raw_assets))
            if len(selected) != len(set(selected)):
                raise ValueError("asset signature repeats one canonical identity")
            hashes = row.get("content_sha256") or {}
            if set(hashes) != set(ROLES) or any(
                len(str(hashes.get(role) or "")) != 64 for role in ROLES
            ):
                raise ValueError("triplet content hashes are incomplete")
            for canonical in selected:
                claimed = next(
                    (
                        item
                        for item in row.get("assets") or []
                        if aliases.get(
                            str(
                                item.get("global_uid")
                                or item.get("obj_id")
                                or item.get("instance_token")
                                or ""
                            )
                        )
                        == canonical
                    ),
                    None,
                )
                if claimed is not None:
                    expected = assets[canonical]
                    claimed_instance = claimed.get("instance_token")
                    claimed_asset_sha256 = claimed.get(
                        "canonical_asset_sha256"
                    ) or claimed.get("asset_sha256")
                    if (
                        claimed_instance is not None
                        and str(claimed_instance) != expected["instance_token"]
                    ) or (
                        claimed_asset_sha256 is not None
                        and str(claimed_asset_sha256)
                        != expected["canonical_asset_sha256"]
                    ):
                        raise ValueError("release asset binding differs from visibility")
            rows.append(
                {
                    "raw": row,
                    "sample_id": sample_id,
                    **parsed,
                    "split": split,
                    "selected_asset_ids": selected,
                    "authority": {
                        "manifest": str(records_path),
                        "manifest_sha256": "",  # filled once per build
                        "line_number": line_number,
                        "row_sha256": canonical_sha256(row),
                    },
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            excluded.append(
                {
                    "sample_id": sample_id or None,
                    "line_number": line_number,
                    "reason": str(error),
                }
            )
    if not rows:
        raise ValueError("accepted authority yielded no strict source rows")
    authority_hash = sha256_file(records_path)
    for row in rows:
        row["authority"]["manifest_sha256"] = authority_hash
    return rows, excluded


def _trajectory_index(
    plan_path: Path,
    wanted: dict[str, set[int]],
    aliases: dict[str, str],
    selected_by_context: dict[str, set[str]],
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    plan_path = plan_path.resolve(strict=True)
    exposures: dict[tuple[str, int, str], dict[str, Any]] = {}
    states: dict[tuple[str, str, int], dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for case_index, case in enumerate(iter_json_array(plan_path, "cases")):
        context = str(case.get("case_id") or "")
        if context not in wanted:
            continue
        scene = str(case.get("scene_name") or "")
        render_context = {
            key: case.get(key)
            for key in (
                "case_id",
                "scene_name",
                "scene_index",
                "scene_token",
                "storm_context_start",
                "storm_context_frames",
                "storm_target_frames",
                "storm_target_offsets",
                "camera_ids",
                "reference_camera_filename",
                "reference_sample_data_token",
                "reference_source_frame",
            )
        }
        if (
            render_context["case_id"] != context
            or render_context["scene_name"] != scene
            or int(render_context["scene_index"] or -1) < 0
            or not render_context["scene_token"]
            or len(render_context["storm_context_frames"] or []) != 4
            or len(render_context["storm_target_frames"] or []) != 4
            or len(render_context["storm_target_offsets"] or []) != 4
            or tuple(map(str, render_context["camera_ids"] or ()))
            != ("1", "0", "2", "4", "5", "3")
        ):
            raise ValueError(f"trajectory render context is incomplete: {context}")
        context_binding = {
            "context_id": context,
            "scene_name": scene,
            "case_index": case_index,
            "plan": str(plan_path),
            "render_context": render_context,
        }
        previous_context = contexts.setdefault(context, context_binding)
        if previous_context != context_binding:
            raise ValueError(f"trajectory plan repeats context: {context}")
        for asset in case.get("assets") or []:
            raw_identity = str(
                asset.get("global_uid")
                or asset.get("obj_id")
                or asset.get("instance_token")
                or ""
            )
            canonical = aliases.get(raw_identity)
            if canonical not in selected_by_context.get(context, set()):
                continue
            trajectory = asset.get("official_trajectory") or {}
            for frame_row in trajectory.get("frames") or []:
                frame = int(frame_row.get("frame_index", -1))
                if frame not in wanted[context]:
                    continue
                camera_states: dict[str, dict[str, Any]] = {}
                for camera, camera_row in (frame_row.get("cameras") or {}).items():
                    camera = str(camera)
                    if camera not in CAMERA_NAMES:
                        continue
                    camera_name = str(camera_row.get("camera_name") or "")
                    timestamp = int(
                        camera_row.get("camera_exposure_timestamp_us") or 0
                    )
                    if camera_name != CAMERA_NAMES[camera] or timestamp <= 0:
                        raise ValueError("trajectory camera exposure is incomplete")
                    exposure = {
                        "camera_id": camera,
                        "camera_name": camera_name,
                        "camera_exposure_timestamp_us": timestamp,
                        "sample_data_timestamp_us": int(
                            camera_row.get("sample_data_timestamp_us") or 0
                        ),
                        "image_path": str(camera_row.get("image_path") or ""),
                    }
                    exposure_key = (context, frame, camera)
                    previous_exposure = exposures.setdefault(exposure_key, exposure)
                    if previous_exposure != exposure:
                        raise ValueError(
                            f"camera exposure changes across assets: {exposure_key}"
                        )
                    camera_states[camera] = {
                        "visible_in_official_projection": bool(
                            camera_row.get("visible")
                        ),
                        "camera_depth_center_m": camera_row.get(
                            "camera_depth_center_m"
                        ),
                        "bbox_xyxy": camera_row.get("bbox_xyxy"),
                    }
                state = {
                    "schema_version": 1,
                    "context_id": context,
                    "scene_name": scene,
                    "frame_index": frame,
                    "global_uid": canonical,
                    "sample_annotation_token": frame_row.get(
                        "sample_annotation_token"
                    ),
                    "translation_global_m": frame_row.get("translation_global_m"),
                    "size_wlh_m": frame_row.get("size_wlh_m"),
                    "rotation_global_quaternion_wxyz": frame_row.get(
                        "rotation_global_quaternion_wxyz"
                    ),
                    "heading_global_xyz": frame_row.get("heading_global_xyz"),
                    "cameras": camera_states,
                }
                state["record_sha256"] = canonical_sha256(state)
                state_key = (context, canonical, frame)
                previous_state = states.setdefault(state_key, state)
                if previous_state != state:
                    raise ValueError(f"asset trajectory changes: {state_key}")
    plan_hash = sha256_file(plan_path)
    for value in contexts.values():
        value["plan_sha256"] = plan_hash
    return exposures, states, contexts


def _resolve_sample_data(
    sample_data_path: Path,
    exposures: dict[tuple[str, int, str], dict[str, Any]],
) -> None:
    required = {
        (int(row["camera_exposure_timestamp_us"]), str(row["camera_name"]))
        for row in exposures.values()
    }
    matched: dict[tuple[int, str], dict[str, Any]] = {}
    for row in iter_json_array(sample_data_path.resolve(strict=True)):
        timestamp = int(row.get("timestamp") or 0)
        filename = str(row.get("filename") or "")
        if timestamp <= 0 or "CAM_" not in filename:
            continue
        camera_name = next(
            (name for name in CAMERA_BY_NAME if f"__{name}__" in filename), None
        )
        key = (timestamp, camera_name) if camera_name else None
        if key not in required:
            continue
        value = {
            "sample_data_token": str(row.get("token") or ""),
            "sample_token": str(row.get("sample_token") or ""),
            "official_filename": filename,
        }
        if any(not value[name] for name in ("sample_data_token", "sample_token")):
            raise ValueError(f"official sample_data identity is incomplete: {key}")
        previous = matched.setdefault(key, value)
        if previous != value:
            raise ValueError(f"official sample_data key is ambiguous: {key}")
    missing = required - set(matched)
    if missing:
        raise ValueError(
            "camera exposures are absent from official sample_data: "
            + ", ".join(map(str, sorted(missing)[:3]))
        )
    for row in exposures.values():
        key = (
            int(row["camera_exposure_timestamp_us"]),
            str(row["camera_name"]),
        )
        row.update(matched[key])


def _metadata_rows(
    metadata_paths: Sequence[Path], wanted_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_path in metadata_paths:
        path = raw_path.resolve(strict=True)
        manifest_hash = sha256_file(path)
        for line_number, row in enumerate(iter_jsonl(path), 1):
            sample_id = str(row.get("sample_id") or "")
            if sample_id in wanted_ids:
                result[sample_id].append(
                    {
                        "row": row,
                        "manifest": str(path),
                        "manifest_sha256": manifest_hash,
                        "line_number": line_number,
                        "row_sha256": canonical_sha256(row),
                    }
                )
    return dict(result)


def _metadata_binding(
    source: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    checkpoint_hashes: dict[str, str],
) -> tuple[dict[str, Any] | None, str]:
    expected_hashes = source["raw"]["content_sha256"]
    mismatched = False
    for candidate in candidates:
        row = candidate["row"]
        if row.get("content_sha256") != expected_hashes:
            mismatched = True
            continue
        if (
            str(row.get("scene_name") or "") != source["scene"]
            or int(row.get("frame_index", -1)) != source["frame_index"]
            or str(row.get("camera_id") or "") != source["camera_id"]
        ):
            raise ValueError(f"metadata identity differs: {source['sample_id']}")
        pair = row.get("pair_contract") or {}
        claimed = {
            "storm": str(pair.get("storm_checkpoint_sha256") or ""),
            "cvac": str(pair.get("cvac_checkpoint_sha256") or ""),
            "dcn": str(pair.get("dcn_checkpoint_sha256") or ""),
        }
        if claimed != checkpoint_hashes:
            raise ValueError(f"metadata checkpoint tuple differs: {source['sample_id']}")
        return candidate, "complete"
    return None, "metadata_hash_mismatch" if mismatched else "release_authority_only"


def _verify_source(source: dict[str, Any]) -> None:
    for role in ROLES:
        path = Path(source["paths"][role])
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"source is not a regular file: {path}")
        if sha256_file(path) != source["content_sha256"][role]:
            raise ValueError(f"source content hash mismatch: {path}")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"source is not an RGB PNG: {path}")
            if image.size != IMAGE_SIZE:
                raise ValueError(f"source dimensions differ: {path}")


def _normalized_sources(
    raw_sources: list[dict[str, Any]],
    source_root: Path,
    metadata: dict[str, list[dict[str, Any]]],
    exposures: dict[tuple[str, int, str], dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    checkpoint_hashes: dict[str, str],
    verify_content: bool,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    source_root = source_root.resolve(strict=True)
    for source in raw_sources:
        sample_id = source["sample_id"]
        try:
            exposure_key = (
                source["context_id"],
                source["frame_index"],
                source["camera_id"],
            )
            exposure = exposures.get(exposure_key)
            if exposure is None:
                raise ValueError("source context/frame is absent from trajectory plan")
            bound_metadata, lineage_status = _metadata_binding(
                source, metadata.get(sample_id, ()), checkpoint_hashes
            )
            if bound_metadata is not None:
                row = bound_metadata["row"]
                if int(row.get("camera_exposure_timestamp_us") or 0) != int(
                    exposure["camera_exposure_timestamp_us"]
                ):
                    raise ValueError("metadata exposure differs from trajectory plan")
            else:
                row = {}
            normalized = {
                "schema_version": 1,
                "source_kind": "accepted_nusc_pair",
                "split": source["split"],
                "sample_id": sample_id,
                "context_id": source["context_id"],
                "scene_name": source["scene"],
                "frame_index": source["frame_index"],
                "camera_id": source["camera_id"],
                "camera_name": CAMERA_NAMES[source["camera_id"]],
                "combination_id": source["combination_id"],
                "selected_asset_ids": list(source["selected_asset_ids"]),
                "assets": [assets[value] for value in source["selected_asset_ids"]],
                "checkpoint_sha256": checkpoint_hashes,
                "exposure": exposure,
                "paths": {
                    role: str(source_root / role / f"{sample_id}.png")
                    for role in ROLES
                },
                "content_sha256": source["raw"]["content_sha256"],
                "lineage_status": lineage_status,
                "authority": source["authority"],
                "production_metadata": (
                    {
                        key: bound_metadata[key]
                        for key in (
                            "manifest",
                            "manifest_sha256",
                            "line_number",
                            "row_sha256",
                        )
                    }
                    if bound_metadata is not None
                    else None
                ),
                "production_record": str(row.get("metadata") or "") or None,
                "quality_gate_pass": (
                    (row.get("quality") or {}).get("quality_gate_pass")
                    if row
                    else None
                ),
            }
            normalized["record_sha256"] = canonical_sha256(normalized)
            output.append(normalized)
        except (KeyError, OSError, TypeError, ValueError) as error:
            excluded.append({"sample_id": sample_id, "reason": str(error)})
    if verify_content and output:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            list(executor.map(_verify_source, output))
    return output, excluded


def _choose(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranks = {"complete": 0, "release_authority_only": 1, "metadata_hash_mismatch": 2}
    return min(
        rows,
        key=lambda row: (ranks.get(str(row.get("lineage_status")), 9), row["sample_id"]),
    )


def _build_groups(
    split: str,
    sources: list[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    visibility: dict[tuple[str, str, int], dict[str, bool]],
    exposures: dict[tuple[str, int, str], dict[str, Any]],
    states: dict[tuple[str, str, int], dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    checkpoint_hashes: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    baselines: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    exact: dict[
        tuple[str, int, str, tuple[str, ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    candidates: set[tuple[str, int, tuple[str, ...]]] = set()
    for row in sources:
        key = (row["context_id"], row["frame_index"], row["camera_id"])
        signature = tuple(row["selected_asset_ids"])
        baselines[key].append(row)
        exact[key + (signature,)].append(row)
        candidates.add((row["context_id"], row["frame_index"], signature))

    groups: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    missing_baselines: dict[tuple[str, int, str], dict[str, Any]] = {}
    missing_visible: dict[
        tuple[str, int, str, tuple[str, ...]], dict[str, Any]
    ] = {}
    for context, frame, selected in sorted(candidates):
        scene = context.split("__", 1)[0]
        reason: str | None = None
        evidence = {
            asset: visibility.get((context, asset, frame)) for asset in selected
        }
        if any(
            not value or set(value) != set(CAMERA_NAMES)
            for value in evidence.values()
        ):
            reason = "context-specific visibility evidence is incomplete"
        trajectory = {
            camera: exposures.get((context, frame, camera)) for camera in CAMERA_NAMES
        }
        if reason is None and any(value is None for value in trajectory.values()):
            reason = "six-camera trajectory exposure is incomplete"
        sample_tokens = {
            str(value.get("sample_token") or "")
            for value in trajectory.values()
            if value is not None
        }
        if reason is None and (len(sample_tokens) != 1 or "" in sample_tokens):
            reason = "six-camera exposures cross official sample_token"
        asset_states = {
            asset: states.get((context, asset, frame)) for asset in selected
        }
        if reason is None and any(value is None for value in asset_states.values()):
            reason = "official asset state is incomplete"
        for camera in CAMERA_NAMES:
            background_rows = baselines.get((context, frame, camera), ())
            background_hashes = {
                (row["content_sha256"]["gt"], row["content_sha256"]["target"])
                for row in background_rows
            }
            if len(background_hashes) > 1:
                reason = "accepted baseline content conflicts"
                break
            visible = tuple(
                asset for asset in selected if evidence.get(asset, {}).get(camera) is True
            )
            pair_rows = exact.get((context, frame, camera, visible), ()) if visible else ()
            if len({row["content_sha256"]["input"] for row in pair_rows}) > 1:
                reason = "accepted visible input content conflicts"
                break
        if reason is not None:
            excluded.append(
                {
                    "split": split,
                    "context_id": context,
                    "scene_name": scene,
                    "frame_index": frame,
                    "selected_asset_ids": list(selected),
                    "reason": reason,
                }
            )
            continue

        union = [assets[value] for value in selected]
        identity = {
            "schema_version": 1,
            "split": split,
            "context_id": context,
            "scene_name": scene,
            "sample_token": next(iter(sample_tokens)),
            "frame_index": frame,
            "asset_union": [
                {
                    "global_uid": row["global_uid"],
                    "obj_id": row["obj_id"],
                    "instance_token": row["instance_token"],
                    "canonical_asset_sha256": row["canonical_asset_sha256"],
                }
                for row in union
            ],
            "checkpoint_sha256": checkpoint_hashes,
        }
        identity_sha = canonical_sha256(identity)
        group_id = f"{split}__{scene}__f{frame:06d}__g-{identity_sha[:16]}"
        views: list[dict[str, Any]] = []
        group_status = "ready"
        for camera, camera_name in CAMERA_RING:
            visible = tuple(
                asset for asset in selected if evidence[asset][camera] is True
            )
            invisible = tuple(asset for asset in selected if asset not in visible)
            background_rows = baselines.get((context, frame, camera), ())
            pair_rows = exact.get((context, frame, camera, visible), ()) if visible else ()
            baseline = _choose(background_rows) if background_rows else None
            pair = _choose(pair_rows) if pair_rows else None
            view_status = "ready"
            if baseline is None:
                view_status = "needs_baseline"
                group_status = "needs_backfill"
                missing_baselines[(context, frame, camera)] = {
                    "context_id": context,
                    "scene_name": scene,
                    "frame_index": frame,
                    "camera_id": camera,
                    "camera_name": camera_name,
                    "exposure": trajectory[camera],
                }
            if visible and pair is None:
                view_status = "needs_baseline_and_visible_pair" if baseline is None else "needs_visible_pair"
                group_status = "needs_backfill"
                missing_visible[(context, frame, camera, visible)] = {
                    "context_id": context,
                    "scene_name": scene,
                    "frame_index": frame,
                    "camera_id": camera,
                    "camera_name": camera_name,
                    "selected_asset_ids": list(visible),
                    "assets": [
                        {
                            **assets[value],
                            "asset_sha256": assets[value][
                                "canonical_asset_sha256"
                            ],
                            # nuScenes stores width,length,height; the frozen
                            # STORM renderer contract consumes length,width,height.
                            "official_dimensions_m": [
                                asset_states[value]["size_wlh_m"][1],
                                asset_states[value]["size_wlh_m"][0],
                                asset_states[value]["size_wlh_m"][2],
                            ],
                        }
                        for value in visible
                    ],
                    "exposure": trajectory[camera],
                }
            views.append(
                {
                    "camera_id": camera,
                    "camera_name": camera_name,
                    "status": view_status,
                    "visible": bool(visible),
                    "no_op": not visible,
                    "no_op_reason": (
                        "no selected asset is usable in this camera according to "
                        "the frozen context visibility manifest"
                        if not visible
                        else None
                    ),
                    "camera_visible_asset_ids": list(visible),
                    "camera_invisible_asset_ids": list(invisible),
                    "asset_state_sha256": {
                        asset: asset_states[asset]["record_sha256"] for asset in visible
                    },
                    "exposure": trajectory[camera],
                    "baseline_source_sample_id": baseline["sample_id"] if baseline else None,
                    "pair_source_sample_id": pair["sample_id"] if pair else None,
                }
            )
        record = {
            "schema_version": 2,
            "status": group_status,
            "group_id": group_id,
            "group_identity": identity,
            "group_identity_sha256": identity_sha,
            "split": split,
            "context_id": context,
            "scene_name": scene,
            "sample_token": identity["sample_token"],
            "frame_index": frame,
            "asset_union": union,
            "asset_state_sha256": {
                asset: asset_states[asset]["record_sha256"] for asset in selected
            },
            "views": views,
        }
        record["record_sha256"] = canonical_sha256(record)
        groups.append(record)

    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in missing_baselines.values():
        by_context[row["context_id"]].append(row)
    baseline_jobs: list[dict[str, Any]] = []
    for context, rows in sorted(by_context.items()):
        binding = contexts[context]
        job = {
            "schema_version": 1,
            "job_kind": "sixcam_baseline_context",
            "job_id": f"baseline-{canonical_sha256([split, context, checkpoint_hashes])[:16]}",
            "split": split,
            "context_id": context,
            "scene_name": binding["scene_name"],
            "trajectory_plan": binding,
            "checkpoint_sha256": checkpoint_hashes,
            "requested_views": sorted(
                rows, key=lambda row: (row["frame_index"], row["camera_id"])
            ),
        }
        job["job_sha256"] = canonical_sha256(job)
        baseline_jobs.append(job)
    visible_jobs: list[dict[str, Any]] = []
    for row in sorted(
        missing_visible.values(),
        key=lambda value: (
            value["context_id"],
            value["frame_index"],
            value["camera_id"],
            value["selected_asset_ids"],
        ),
    ):
        sample_id = (
            f"{row['context_id']}__cmb-"
            f"{canonical_sha256(row['selected_asset_ids'])[:12]}__"
            f"f{int(row['frame_index']):03d}_c{row['camera_id']}"
        )
        job = {
            "schema_version": 1,
            "job_kind": "sixcam_visible_backfill",
            "split": split,
            **row,
            "sample_id": sample_id,
            "window_id": row["context_id"],
            "selected_obj_ids": [asset["obj_id"] for asset in row["assets"]],
            "checkpoint_sha256": checkpoint_hashes,
        }
        job["job_id"] = f"visible-{canonical_sha256(job)[:16]}"
        job["job_sha256"] = canonical_sha256(job)
        visible_jobs.append(job)
    return groups, excluded, baseline_jobs, visible_jobs


def build_strict_sixcam_index(
    *,
    split: str,
    source_root: Path,
    records_path: Path,
    metadata_paths: Sequence[Path],
    visibility_manifest: Path,
    trajectory_plan: Path,
    sample_data_table: Path,
    official_scenes: Path,
    render_contract: Path,
    output_root: Path,
    verify_content: bool = True,
    workers: int = 32,
) -> dict[str, Any]:
    """Create strict group and deficit manifests without rendering or publishing."""

    if split not in {"train", "val"} or workers < 1:
        raise ValueError("strict index split/workers are invalid")
    scene_names = _official_scenes(official_scenes, split)
    aliases, assets, visibility, visibility_contexts = load_context_visibility(
        visibility_manifest
    )
    checkpoint_hashes, checkpoint_paths = _checkpoint_contract(
        render_contract, verify_content
    )
    raw_sources, source_exclusions = _release_rows(
        records_path, split, scene_names, aliases, assets
    )
    wanted: dict[str, set[int]] = defaultdict(set)
    selected_by_context: dict[str, set[str]] = defaultdict(set)
    for row in raw_sources:
        wanted[row["context_id"]].add(row["frame_index"])
        selected_by_context[row["context_id"]].update(row["selected_asset_ids"])
    exposures, states, trajectory_contexts = _trajectory_index(
        trajectory_plan, dict(wanted), aliases, dict(selected_by_context)
    )
    _resolve_sample_data(sample_data_table, exposures)
    metadata = _metadata_rows(metadata_paths, {row["sample_id"] for row in raw_sources})
    sources, normalization_exclusions = _normalized_sources(
        raw_sources,
        source_root,
        metadata,
        exposures,
        assets,
        checkpoint_hashes,
        verify_content,
        workers,
    )
    context_bindings: dict[str, dict[str, Any]] = {}
    for context, value in trajectory_contexts.items():
        visibility_value = visibility_contexts.get(context)
        if visibility_value is None or visibility_value["scene_name"] != value["scene_name"]:
            continue
        context_bindings[context] = {
            **value,
            "visibility_manifest": str(visibility_manifest.resolve(strict=True)),
            "visibility_manifest_sha256": sha256_file(
                visibility_manifest.resolve(strict=True)
            ),
        }
    groups, group_exclusions, baseline_jobs, visible_jobs = _build_groups(
        split,
        sources,
        assets,
        visibility,
        exposures,
        states,
        context_bindings,
        checkpoint_hashes,
    )
    ready_groups = [row for row in groups if row["status"] == "ready"]
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "source_index": output_root / "source_index.jsonl",
        "asset_states": output_root / "asset_states.jsonl",
        "groups": output_root / "groups.jsonl",
        "ready_groups": output_root / "ready_groups.jsonl",
        "baseline_jobs": output_root / "baseline_jobs.jsonl",
        "visible_backfill_jobs": output_root / "visible_backfill_jobs.jsonl",
        "excluded_groups": output_root / "excluded_groups.jsonl",
        "source_exclusions": output_root / "source_exclusions.jsonl",
    }
    atomic_jsonl(manifests["source_index"], sources)
    atomic_jsonl(
        manifests["asset_states"],
        (states[key] for key in sorted(states)),
    )
    atomic_jsonl(manifests["groups"], groups)
    atomic_jsonl(manifests["ready_groups"], ready_groups)
    atomic_jsonl(manifests["baseline_jobs"], baseline_jobs)
    atomic_jsonl(manifests["visible_backfill_jobs"], visible_jobs)
    atomic_jsonl(manifests["excluded_groups"], group_exclusions)
    atomic_jsonl(
        manifests["source_exclusions"],
        [*source_exclusions, *normalization_exclusions],
    )
    status_counts = Counter(row["status"] for row in groups)
    lineage_counts = Counter(row["lineage_status"] for row in sources)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "operation": "strict_sixcam_read_only_index",
        "split": split,
        "content_verification": {
            "performed": bool(verify_content),
            "policy": (
                "regular_file_sha256_decode_rgb_png_512x288"
                if verify_content
                else "development_skip_recorded_hashes_only"
            ),
        },
        "source_count": len(sources),
        "source_exclusion_count": len(source_exclusions)
        + len(normalization_exclusions),
        "source_lineage_counts": dict(sorted(lineage_counts.items())),
        "candidate_group_count": len(groups),
        "ready_group_count": len(ready_groups),
        "needs_backfill_group_count": status_counts["needs_backfill"],
        "excluded_group_count": len(group_exclusions),
        "baseline_context_job_count": len(baseline_jobs),
        "missing_baseline_view_count": sum(
            len(row["requested_views"]) for row in baseline_jobs
        ),
        "missing_visible_view_count": len(visible_jobs),
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_paths": checkpoint_paths,
        "inputs": {
            "source_root": str(source_root.resolve(strict=True)),
            "records": str(records_path.resolve(strict=True)),
            "records_sha256": sha256_file(records_path.resolve(strict=True)),
            "metadata": [str(path.resolve(strict=True)) for path in metadata_paths],
            "visibility_manifest": str(visibility_manifest.resolve(strict=True)),
            "visibility_manifest_sha256": sha256_file(
                visibility_manifest.resolve(strict=True)
            ),
            "trajectory_plan": str(trajectory_plan.resolve(strict=True)),
            "trajectory_plan_sha256": sha256_file(trajectory_plan.resolve(strict=True)),
            "sample_data_table": str(sample_data_table.resolve(strict=True)),
            "sample_data_table_sha256": sha256_file(
                sample_data_table.resolve(strict=True)
            ),
            "official_scenes": str(official_scenes.resolve(strict=True)),
            "official_scenes_sha256": sha256_file(official_scenes.resolve(strict=True)),
            "render_contract": str(render_contract.resolve(strict=True)),
            "render_contract_sha256": sha256_file(render_contract.resolve(strict=True)),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in manifests.items()
        },
    }
    summary["summary_payload_sha256"] = canonical_sha256(summary)
    atomic_json(output_root / "summary.json", summary)
    return summary
