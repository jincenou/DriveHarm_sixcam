"""Resume-safe strict six-camera render, compose, publish, and audit controller."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Sequence

from .compose import compose_results
from .contracts import atomic_json, read_json
from .render import render_shards
from .sixcam import audit_strict_sixcam_release, publish_strict_sixcam_release


def _gpus(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU list is empty or duplicated")
    return result


def _state(path: Path, stage: str, stages: dict[str, Any]) -> None:
    atomic_json(
        path,
        {
            "schema_version": 1,
            "status": "running" if stage != "complete" else "complete",
            "stage": stage,
            "stages": stages,
        },
    )


async def run_strict_sixcam_production(
    *,
    train_index: Path,
    val_index: Path,
    train_profile: Path,
    val_profile: Path,
    baseline_python: Path,
    visible_python: Path,
    render_contract: Path,
    work_root: Path,
    destination: Path,
    receipt_root: Path,
    gpus: Sequence[int] = (0, 1, 2, 3),
    workers_per_gpu: int = 1,
    shards_per_worker: int = 4,
    visible_shards_per_worker: int = 1,
    publish_workers: int = 32,
) -> dict[str, Any]:
    """Run all full-production stages; bounded job failures are quarantined."""

    if tuple(gpus) != tuple(sorted(set(gpus))) or not gpus:
        raise ValueError("production GPU list must be nonempty, sorted, and unique")
    indices = {"train": train_index.resolve(strict=True), "val": val_index.resolve(strict=True)}
    profiles = {"train": train_profile.resolve(strict=True), "val": val_profile.resolve(strict=True)}
    for split, root in indices.items():
        summary = read_json((root / "summary.json").resolve(strict=True))
        if (
            summary.get("split") != split
            or summary.get("status") != "complete"
            or (summary.get("content_verification") or {}).get("performed")
            is not True
        ):
            raise ValueError(f"production requires a content-verified {split} index")
    baseline_python = baseline_python.resolve(strict=True)
    visible_python = visible_python.resolve(strict=True)
    render_contract = render_contract.resolve(strict=True)
    work_root.mkdir(parents=True, exist_ok=True)
    state_path = work_root / "production_state.json"
    stages: dict[str, Any] = {}
    _state(state_path, "starting", stages)

    baseline_results: list[Path] = []
    for split in ("train", "val"):
        stage = f"{split}_baseline"
        _state(state_path, stage, stages)
        result = await render_shards(
            indices[split] / "baseline_jobs.jsonl",
            work_root / stage,
            baseline_python,
            render_contract,
            renderer_args=("-m", "driveharm.storm_baseline", "--profile", str(profiles[split])),
            gpus=gpus,
            workers_per_gpu=workers_per_gpu,
            shards_per_worker=shards_per_worker,
        )
        stages[stage] = result
        baseline_results.append(Path(result["results"]))
        _state(state_path, f"{stage}_complete", stages)

    visible_records: list[Path] = []
    for split in ("train", "val"):
        stage = f"{split}_visible"
        _state(state_path, stage, stages)
        result = await render_shards(
            indices[split] / "visible_backfill_jobs.jsonl",
            work_root / stage,
            visible_python,
            render_contract,
            renderer_args=("-m", "driveharm.storm_adapter", "--profile", str(profiles[split])),
            gpus=gpus,
            workers_per_gpu=workers_per_gpu,
            shards_per_worker=visible_shards_per_worker,
        )
        stages[stage] = result
        composed = compose_results(
            Path(result["results"]), work_root / f"{split}_visible_composed"
        )
        stages[f"{split}_visible_compose"] = composed
        visible_records.append(Path(composed["records"]))
        _state(state_path, f"{stage}_complete", stages)

    _state(state_path, "publish", stages)
    published = publish_strict_sixcam_release(
        index_roots=[indices["train"], indices["val"]],
        destination=destination,
        receipt_root=receipt_root,
        baseline_result_paths=baseline_results,
        visible_result_paths=visible_records,
        materialize="hardlink",
        replace=False,
        workers=publish_workers,
    )
    stages["publish"] = published
    _state(state_path, "independent_audit", stages)
    audited = audit_strict_sixcam_release(
        destination, receipt_root / "independent_audit", publish_workers
    )
    stages["independent_audit"] = audited
    if audited.get("status") != "pass":
        raise RuntimeError("published strict six-camera release failed independent audit")
    _state(state_path, "complete", stages)
    return {"status": "complete", "stages": stages}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--train-index", type=Path, required=True)
    value.add_argument("--val-index", type=Path, required=True)
    value.add_argument("--train-profile", type=Path, required=True)
    value.add_argument("--val-profile", type=Path, required=True)
    value.add_argument("--baseline-python", type=Path, required=True)
    value.add_argument("--visible-python", type=Path, required=True)
    value.add_argument("--render-contract", type=Path, required=True)
    value.add_argument("--work-root", type=Path, required=True)
    value.add_argument("--destination", type=Path, required=True)
    value.add_argument("--receipt-root", type=Path, required=True)
    value.add_argument("--gpus", type=_gpus, default=(0, 1, 2, 3))
    value.add_argument("--workers-per-gpu", type=int, default=1)
    value.add_argument("--shards-per-worker", type=int, default=4)
    value.add_argument("--visible-shards-per-worker", type=int, default=1)
    value.add_argument("--publish-workers", type=int, default=32)
    return value


def main() -> int:
    args = parser().parse_args()
    result = asyncio.run(
        run_strict_sixcam_production(
            train_index=args.train_index,
            val_index=args.val_index,
            train_profile=args.train_profile,
            val_profile=args.val_profile,
            baseline_python=args.baseline_python,
            visible_python=args.visible_python,
            render_contract=args.render_contract,
            work_root=args.work_root,
            destination=args.destination,
            receipt_root=args.receipt_root,
            gpus=args.gpus,
            workers_per_gpu=args.workers_per_gpu,
            shards_per_worker=args.shards_per_worker,
            visible_shards_per_worker=args.visible_shards_per_worker,
            publish_workers=args.publish_workers,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
