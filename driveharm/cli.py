"""Command-line interface and end-to-end DriveHarm controller."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from .audit import audit_release
from .compose import compose_results
from .contracts import atomic_json, atomic_jsonl, indexed_rows, iter_jsonl
from .planning import build_plan
from .release import publish_release, quarantine_triplets
from .render import render_shards
from .review import review_manifest
from .sixcam import (
    audit_sixcam_release,
    audit_strict_sixcam_release,
    build_strict_sixcam_review_sheets,
    build_sixcam_release,
    publish_strict_sixcam_release,
)
from .sixcam_index import build_strict_sixcam_index
from .sixcam_pilot import build_strict_sixcam_pilot_plan


def _gpus(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU list is empty or duplicated")
    return result


def _key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"API key environment variable is empty: {name}")
    return value


def _accepted_jobs(jobs: Path, decisions: Path, output: Path) -> None:
    reviewed = indexed_rows(decisions, "sample_id")
    rows = [
        row
        for row in iter_jsonl(jobs)
        if all(
            reviewed.get(str(asset["asset_id"]), {}).get("hard_failure") is False
            for asset in row["assets"]
        )
    ]
    if not rows:
        raise RuntimeError("identity review accepted no jobs")
    atomic_jsonl(output, rows)


def _asset_review_manifest(jobs: Path, output: Path) -> None:
    assets: dict[str, dict[str, Any]] = {}
    for job in iter_jsonl(jobs):
        if len(job.get("assets") or []) != 1:
            continue
        asset = job["assets"][0]
        asset_id = str(asset["asset_id"])
        row = {
            "sample_id": asset_id,
            "review_images": job.get("review_images") or {},
            "review_context": {"stage": "asset_identity", **asset},
        }
        if asset_id in assets and assets[asset_id] != row:
            raise ValueError(f"identity evidence differs across jobs: {asset_id}")
        assets[asset_id] = row
    if not assets or any(not row["review_images"] for row in assets.values()):
        raise ValueError("every exact asset requires identity review images")
    atomic_jsonl(output, (assets[key] for key in sorted(assets)))


def _pair_review_manifest(records: Path, output: Path) -> None:
    rows = []
    for record in iter_jsonl(records):
        rows.append(
            {
                "sample_id": record["sample_id"],
                "review_images": record["paths"],
                "review_context": {
                    "stage": "rendered_triplet",
                    "camera_id": record["camera_id"],
                    "selected_obj_ids": record["selected_obj_ids"],
                    "instruction": "gt, input and target are shown as separate labeled images",
                },
            }
        )
    atomic_jsonl(output, rows)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        args.asset_post,
        args.observations,
        work / "01_plan",
        args.identity_manifest,
        maximum_combinations_per_window=args.maximum_combinations_per_window,
        combination_seed=args.combination_seed,
    )
    identity_manifest = work / "02_identity_review/manifest.jsonl"
    identity_manifest.parent.mkdir(parents=True, exist_ok=True)
    _asset_review_manifest(Path(plan["jobs"]), identity_manifest)
    identity_review = await review_manifest(
        identity_manifest,
        work / "02_identity_review",
        args.base_url,
        _key(args.api_key_env),
        args.model,
        args.review_concurrency,
        args.timeout,
    )
    reviewed_jobs = work / "03_render/reviewed_jobs.jsonl"
    reviewed_jobs.parent.mkdir(parents=True, exist_ok=True)
    _accepted_jobs(
        Path(plan["jobs"]), Path(identity_review["decisions"]), reviewed_jobs
    )
    rendered = await render_shards(
        reviewed_jobs,
        work / "03_render",
        args.renderer,
        args.render_contract,
        args.renderer_arg,
        args.gpus,
        args.workers_per_gpu,
        args.shards_per_worker,
    )
    composed = compose_results(Path(rendered["results"]), work / "04_composed")
    pair_manifest = work / "05_pair_review/manifest.jsonl"
    pair_manifest.parent.mkdir(parents=True, exist_ok=True)
    _pair_review_manifest(Path(composed["records"]), pair_manifest)
    pair_review = await review_manifest(
        pair_manifest,
        work / "05_pair_review",
        args.base_url,
        _key(args.api_key_env),
        args.model,
        args.review_concurrency,
        args.timeout,
    )
    audited = audit_release(
        work / "04_composed",
        Path(composed["records"]),
        work / "06_audit",
        Path(pair_review["decisions"]),
        args.audit_workers,
    )
    published = publish_release(
        [work / "04_composed"],
        [Path(audited["accepted_records"])],
        args.destination,
        work / "07_release",
        args.materialize,
        args.replace,
        args.publish_workers,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "plan": plan,
        "identity_review": identity_review,
        "render": rendered,
        "compose": composed,
        "pair_review": pair_review,
        "audit": audited,
        "release": published,
    }
    atomic_json(work / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="driveharm")
    sub = root.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--asset-post", type=Path, required=True)
    plan.add_argument("--observations", type=Path, required=True)
    plan.add_argument("--identity-manifest", type=Path)
    plan.add_argument("--maximum-combinations-per-window", type=int, default=256)
    plan.add_argument("--combination-seed", type=int, default=20260818)
    plan.add_argument("--output-root", type=Path, required=True)

    review = sub.add_parser("review")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--output-root", type=Path, required=True)
    review.add_argument("--base-url", required=True)
    review.add_argument("--api-key-env", default="OPENAI_API_KEY")
    review.add_argument("--model", required=True)
    review.add_argument("--concurrency", type=int, default=16)
    review.add_argument("--timeout", type=float, default=180)

    render = sub.add_parser("render")
    render.add_argument("--jobs", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--renderer", type=Path, required=True)
    render.add_argument("--render-contract", type=Path, required=True)
    render.add_argument("--renderer-arg", action="append", default=[])
    render.add_argument("--gpus", type=_gpus, default=tuple(range(8)))
    render.add_argument("--workers-per-gpu", type=int, default=2)
    render.add_argument("--shards-per-worker", type=int, default=4)

    compose = sub.add_parser("compose")
    compose.add_argument("--results", type=Path, required=True)
    compose.add_argument("--output-root", type=Path, required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--dataset-root", type=Path, required=True)
    audit.add_argument("--records", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--visual-decisions", type=Path)
    audit.add_argument("--workers", type=int, default=48)

    quarantine = sub.add_parser("quarantine")
    quarantine.add_argument("--dataset-root", type=Path, required=True)
    quarantine.add_argument("--candidate-ids", type=Path, required=True)
    quarantine.add_argument("--quarantine-root", type=Path, required=True)
    quarantine.add_argument("--receipt", type=Path, required=True)

    release = sub.add_parser("release")
    release.add_argument("--source-root", type=Path, action="append", required=True)
    release.add_argument(
        "--accepted-records", type=Path, action="append", required=True
    )
    release.add_argument("--destination", type=Path, required=True)
    release.add_argument("--receipt-root", type=Path, required=True)
    release.add_argument(
        "--materialize", choices=("hardlink", "copy"), default="hardlink"
    )
    release.add_argument("--replace", action="store_true")
    release.add_argument("--workers", type=int, default=32)

    ring = sub.add_parser("sixcam-release")
    ring.add_argument("--source-root", type=Path, required=True)
    ring.add_argument("--records", type=Path, required=True)
    ring.add_argument("--visibility-manifest", type=Path, required=True)
    ring.add_argument("--destination", type=Path, required=True)
    ring.add_argument("--receipt-root", type=Path, required=True)
    ring.add_argument("--maximum-groups", type=int, default=0)
    ring.add_argument("--scene", action="append", default=[])
    ring.add_argument("--category", action="append", default=[])
    ring.add_argument("--materialize", choices=("hardlink", "copy"), default="hardlink")
    ring.add_argument("--replace", action="store_true")
    ring.add_argument("--workers", type=int, default=32)

    ring_audit = sub.add_parser("sixcam-audit")
    ring_audit.add_argument("--dataset-root", type=Path, required=True)
    ring_audit.add_argument("--groups", type=Path, required=True)
    ring_audit.add_argument("--source-records", type=Path, required=True)
    ring_audit.add_argument("--visibility-manifest", type=Path, required=True)
    ring_audit.add_argument("--output-root", type=Path, required=True)
    ring_audit.add_argument("--workers", type=int, default=32)

    strict_index = sub.add_parser("sixcam-index")
    strict_index.add_argument("--split", choices=("train", "val"), required=True)
    strict_index.add_argument("--source-root", type=Path, required=True)
    strict_index.add_argument("--records", type=Path, required=True)
    strict_index.add_argument("--metadata", type=Path, action="append", default=[])
    strict_index.add_argument("--visibility-manifest", type=Path, required=True)
    strict_index.add_argument("--trajectory-plan", type=Path, required=True)
    strict_index.add_argument("--sample-data-table", type=Path, required=True)
    strict_index.add_argument("--official-scenes", type=Path, required=True)
    strict_index.add_argument("--render-contract", type=Path, required=True)
    strict_index.add_argument("--output-root", type=Path, required=True)
    strict_index.add_argument("--workers", type=int, default=32)
    strict_index.add_argument(
        "--skip-content-verification",
        action="store_true",
        help="development-only: trust recorded source/checkpoint hashes",
    )

    strict_release = sub.add_parser("sixcam-strict-release")
    strict_release.add_argument(
        "--index-root", type=Path, action="append", required=True
    )
    strict_release.add_argument(
        "--baseline-results", type=Path, action="append", default=[]
    )
    strict_release.add_argument(
        "--visible-results", type=Path, action="append", default=[]
    )
    strict_release.add_argument(
        "--receipt", type=Path, action="append", default=[]
    )
    strict_release.add_argument(
        "--report", type=Path, action="append", default=[]
    )
    strict_release.add_argument("--production-command")
    strict_release.add_argument("--selection", type=Path)
    strict_release.add_argument("--destination", type=Path, required=True)
    strict_release.add_argument("--receipt-root", type=Path, required=True)
    strict_release.add_argument(
        "--materialize", choices=("hardlink", "copy"), default="hardlink"
    )
    strict_release.add_argument("--replace", action="store_true")
    strict_release.add_argument("--workers", type=int, default=32)

    strict_audit = sub.add_parser("sixcam-strict-audit")
    strict_audit.add_argument("--dataset-root", type=Path, required=True)
    strict_audit.add_argument("--output-root", type=Path, required=True)
    strict_audit.add_argument("--workers", type=int, default=32)

    strict_pilot = sub.add_parser("sixcam-pilot-plan")
    strict_pilot.add_argument(
        "--index-root", type=Path, action="append", required=True
    )
    strict_pilot.add_argument("--output-root", type=Path, required=True)
    strict_pilot.add_argument("--train-count", type=int, default=8)
    strict_pilot.add_argument("--val-count", type=int, default=4)
    strict_pilot.add_argument("--direct-reuse-count", type=int, default=2)

    strict_sheets = sub.add_parser("sixcam-review-sheets")
    strict_sheets.add_argument("--dataset-root", type=Path, required=True)
    strict_sheets.add_argument("--output-root", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--asset-post", type=Path, required=True)
    run.add_argument("--observations", type=Path, required=True)
    run.add_argument("--identity-manifest", type=Path, required=True)
    run.add_argument("--maximum-combinations-per-window", type=int, default=256)
    run.add_argument("--combination-seed", type=int, default=20260818)
    run.add_argument("--work-root", type=Path, required=True)
    run.add_argument("--destination", type=Path, required=True)
    run.add_argument("--renderer", type=Path, required=True)
    run.add_argument("--render-contract", type=Path, required=True)
    run.add_argument("--renderer-arg", action="append", default=[])
    run.add_argument("--gpus", type=_gpus, default=tuple(range(8)))
    run.add_argument("--workers-per-gpu", type=int, default=2)
    run.add_argument("--shards-per-worker", type=int, default=4)
    run.add_argument("--base-url", required=True)
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--model", required=True)
    run.add_argument("--review-concurrency", type=int, default=16)
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--audit-workers", type=int, default=48)
    run.add_argument("--materialize", choices=("hardlink", "copy"), default="hardlink")
    run.add_argument("--replace", action="store_true")
    run.add_argument("--publish-workers", type=int, default=32)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "plan":
        result = build_plan(
            args.asset_post,
            args.observations,
            args.output_root,
            args.identity_manifest,
            maximum_combinations_per_window=args.maximum_combinations_per_window,
            combination_seed=args.combination_seed,
        )
    elif args.command == "review":
        result = asyncio.run(
            review_manifest(
                args.manifest,
                args.output_root,
                args.base_url,
                _key(args.api_key_env),
                args.model,
                args.concurrency,
                args.timeout,
            )
        )
    elif args.command == "render":
        result = asyncio.run(
            render_shards(
                args.jobs,
                args.output_root,
                args.renderer,
                args.render_contract,
                args.renderer_arg,
                args.gpus,
                args.workers_per_gpu,
                args.shards_per_worker,
            )
        )
    elif args.command == "compose":
        result = compose_results(args.results, args.output_root)
    elif args.command == "audit":
        result = audit_release(
            args.dataset_root,
            args.records,
            args.output_root,
            args.visual_decisions,
            args.workers,
        )
    elif args.command == "quarantine":
        result = quarantine_triplets(
            args.dataset_root,
            args.candidate_ids,
            args.quarantine_root,
            args.receipt,
        )
    elif args.command == "release":
        result = publish_release(
            args.source_root,
            args.accepted_records,
            args.destination,
            args.receipt_root,
            args.materialize,
            args.replace,
            args.workers,
        )
    elif args.command == "sixcam-release":
        result = build_sixcam_release(
            args.source_root,
            args.records,
            args.visibility_manifest,
            args.destination,
            args.receipt_root,
            args.maximum_groups,
            args.scene,
            args.category,
            args.materialize,
            args.replace,
            args.workers,
        )
    elif args.command == "sixcam-audit":
        result = audit_sixcam_release(
            args.dataset_root,
            args.groups,
            args.source_records,
            args.visibility_manifest,
            args.output_root,
            args.workers,
        )
    elif args.command == "sixcam-index":
        result = build_strict_sixcam_index(
            split=args.split,
            source_root=args.source_root,
            records_path=args.records,
            metadata_paths=args.metadata,
            visibility_manifest=args.visibility_manifest,
            trajectory_plan=args.trajectory_plan,
            sample_data_table=args.sample_data_table,
            official_scenes=args.official_scenes,
            render_contract=args.render_contract,
            output_root=args.output_root,
            verify_content=not args.skip_content_verification,
            workers=args.workers,
        )
    elif args.command == "sixcam-strict-release":
        result = publish_strict_sixcam_release(
            index_roots=args.index_root,
            destination=args.destination,
            receipt_root=args.receipt_root,
            baseline_result_paths=args.baseline_results,
            visible_result_paths=args.visible_results,
            receipt_paths=args.receipt,
            report_paths=args.report,
            production_command=args.production_command,
            selection_path=args.selection,
            materialize=args.materialize,
            replace=args.replace,
            workers=args.workers,
        )
    elif args.command == "sixcam-strict-audit":
        result = audit_strict_sixcam_release(
            args.dataset_root, args.output_root, args.workers
        )
    elif args.command == "sixcam-pilot-plan":
        result = build_strict_sixcam_pilot_plan(
            args.index_root,
            args.output_root,
            args.train_count,
            args.val_count,
            args.direct_reuse_count,
        )
    elif args.command == "sixcam-review-sheets":
        result = build_strict_sixcam_review_sheets(
            args.dataset_root, args.output_root
        )
    else:
        result = asyncio.run(_run(args))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
