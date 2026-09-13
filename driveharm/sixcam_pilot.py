"""Deterministically select and materialize a strict six-camera pilot plan."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .contracts import (
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    iter_jsonl,
    sha256_file,
)
from .sixcam import _strict_indices


def _features(group: dict[str, Any]) -> set[str]:
    views = group.get("views") or []
    visible = [view for view in views if view.get("visible") is True]
    return {
        f"status:{group.get('status')}",
        "asset_count:multi" if len(group.get("asset_union") or []) > 1 else "asset_count:single",
        "visible_cameras:multi" if len(visible) > 1 else "visible_cameras:single",
        "no_op:four_plus" if len(views) - len(visible) >= 4 else "no_op:fewer",
        "visible_backfill:yes"
        if any("visible_pair" in str(view.get("status") or "") for view in views)
        else "visible_backfill:no",
        *(f"visible_camera:{view.get('camera_id')}" for view in visible),
    }


def _diverse_groups(
    pool: Sequence[dict[str, Any]],
    count: int,
    selected: Sequence[dict[str, Any]] = (),
    require_visible_backfill: bool = False,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("pilot selection count cannot be negative")
    chosen = list(selected)
    output: list[dict[str, Any]] = []
    remaining = {
        str(group["group_id"]): group
        for group in pool
        if str(group.get("group_id") or "")
        not in {str(row["group_id"]) for row in chosen}
    }
    if len(remaining) < count:
        raise ValueError("pilot candidate pool is smaller than the requested count")
    for ordinal in range(count):
        observed_features = set().union(*(_features(row) for row in chosen)) if chosen else set()
        observed_scenes = {str(row.get("scene_name") or "") for row in chosen}
        observed_contexts = {str(row.get("context_id") or "") for row in chosen}

        def rank(group: dict[str, Any]) -> tuple[int, str]:
            features = _features(group)
            score = 25 * len(features - observed_features)
            score += 120 * (str(group.get("scene_name") or "") not in observed_scenes)
            score += 50 * (str(group.get("context_id") or "") not in observed_contexts)
            if (
                require_visible_backfill
                and ordinal == 0
                and "visible_backfill:yes" in features
            ):
                score += 10_000
            return score, str(group["group_id"])

        best = max(remaining.values(), key=rank)
        if (
            require_visible_backfill
            and ordinal == 0
            and "visible_backfill:yes" not in _features(best)
        ):
            raise ValueError("pilot pool has no required visible-backfill case")
        output.append(best)
        chosen.append(best)
        remaining.pop(str(best["group_id"]))
    return output


def build_strict_sixcam_pilot_plan(
    index_roots: Sequence[Path],
    output_root: Path,
    train_count: int = 8,
    val_count: int = 4,
    direct_reuse_count: int = 2,
) -> dict[str, Any]:
    """Select diverse groups and extract only their required render jobs."""

    if (
        train_count < direct_reuse_count
        or train_count < 1
        or val_count < 1
        or direct_reuse_count < 1
    ):
        raise ValueError("strict pilot counts are invalid")
    groups, _sources, baseline_by_view, visible_by_id = _strict_indices(index_roots)
    train = [group for group in groups if group.get("split") == "train"]
    val = [group for group in groups if group.get("split") == "val"]
    ready = [group for group in train if group.get("status") == "ready"]
    backfill = [group for group in train if group.get("status") == "needs_backfill"]

    selected: list[dict[str, Any]] = []
    selected.extend(_diverse_groups(ready, direct_reuse_count, selected))
    selected.extend(
        _diverse_groups(
            backfill,
            train_count - direct_reuse_count,
            selected,
            require_visible_backfill=True,
        )
    )
    selected.extend(
        _diverse_groups(
            val,
            val_count,
            selected,
            require_visible_backfill=True,
        )
    )

    baseline_jobs: dict[str, dict[str, Any]] = {}
    visible_lookup = {
        (
            job["split"],
            job["context_id"],
            int(job["frame_index"]),
            str(job["camera_id"]),
            tuple(job["selected_asset_ids"]),
        ): job
        for job in visible_by_id.values()
    }
    visible_jobs: dict[str, dict[str, Any]] = {}
    for group in selected:
        for view in group.get("views") or []:
            camera = str(view["camera_id"])
            if not view.get("baseline_source_sample_id"):
                key = (
                    str(group["split"]),
                    str(group["context_id"]),
                    int(group["frame_index"]),
                    camera,
                )
                job = baseline_by_view.get(key)
                if job is None:
                    raise ValueError(f"pilot baseline job is missing: {key}")
                baseline_jobs[str(job["job_id"])] = job
            if view.get("visible") is True and not view.get("pair_source_sample_id"):
                key = (
                    str(group["split"]),
                    str(group["context_id"]),
                    int(group["frame_index"]),
                    camera,
                    tuple(view.get("camera_visible_asset_ids") or []),
                )
                job = visible_lookup.get(key)
                if job is None:
                    raise ValueError(f"pilot visible job is missing: {key}")
                visible_jobs[str(job["sample_id"])] = job

    rejected_fixtures: list[dict[str, Any]] = []
    for root in index_roots:
        split = str(root.name)
        for row in iter_jsonl(root.resolve(strict=True) / "excluded_groups.jsonl"):
            if "cross official sample_token" in str(row.get("reason") or ""):
                rejected_fixtures.append({"split": split, **row})
                break
    if {row["split"] for row in rejected_fixtures} != {"train", "val"}:
        raise ValueError("pilot has no cross-sample-token rejection fixture per split")

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": output_root / "selection.jsonl",
        "groups": output_root / "groups.jsonl",
        "train_baseline_jobs": output_root / "train_baseline_jobs.jsonl",
        "val_baseline_jobs": output_root / "val_baseline_jobs.jsonl",
        "train_visible_jobs": output_root / "train_visible_jobs.jsonl",
        "val_visible_jobs": output_root / "val_visible_jobs.jsonl",
        "rejected_fixtures": output_root / "cross_token_rejections.jsonl",
    }
    atomic_jsonl(
        paths["selection"], ({"group_id": group["group_id"]} for group in selected)
    )
    atomic_jsonl(paths["groups"], selected)
    for split in ("train", "val"):
        atomic_jsonl(
            paths[f"{split}_baseline_jobs"],
            (
                baseline_jobs[key]
                for key in sorted(baseline_jobs)
                if baseline_jobs[key]["split"] == split
            ),
        )
        atomic_jsonl(
            paths[f"{split}_visible_jobs"],
            (
                visible_jobs[key]
                for key in sorted(visible_jobs)
                if visible_jobs[key]["split"] == split
            ),
        )
    atomic_jsonl(paths["rejected_fixtures"], rejected_fixtures)
    feature_counts = Counter(
        feature for group in selected for feature in _features(group)
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "operation": "strict_sixcam_pilot_plan",
        "group_count": len(selected),
        "split_counts": dict(Counter(group["split"] for group in selected)),
        "direct_reuse_group_count": sum(
            group.get("status") == "ready" for group in selected
        ),
        "backfill_group_count": sum(
            group.get("status") == "needs_backfill" for group in selected
        ),
        "baseline_context_job_count": len(baseline_jobs),
        "visible_backfill_job_count": len(visible_jobs),
        "baseline_context_job_split_counts": dict(
            Counter(job["split"] for job in baseline_jobs.values())
        ),
        "visible_backfill_job_split_counts": dict(
            Counter(job["split"] for job in visible_jobs.values())
        ),
        "feature_counts": dict(sorted(feature_counts.items())),
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    summary["summary_payload_sha256"] = canonical_sha256(summary)
    atomic_json(output_root / "summary.json", summary)
    return summary
