"""Asynchronous multi-GPU dispatcher for a company STORM render executable."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Sequence

from .contracts import (
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    iter_jsonl,
    read_json,
    sha256_file,
)


def _job_identity(row: dict[str, Any]) -> str:
    value = str(row.get("sample_id") or row.get("job_id") or "")
    if not value:
        raise ValueError("render job/result has no stable identity")
    return value


async def render_shards(
    jobs_path: Path,
    output_root: Path,
    renderer: Path,
    render_contract: Path,
    renderer_args: Sequence[str] = (),
    gpus: Sequence[int] = tuple(range(8)),
    workers_per_gpu: int = 2,
    shards_per_worker: int = 4,
) -> dict[str, Any]:
    if (
        not gpus
        or len(set(gpus)) != len(gpus)
        or workers_per_gpu < 1
        or shards_per_worker < 1
    ):
        raise ValueError("GPU configuration is invalid")
    renderer = renderer.resolve(strict=True)
    if not os.access(renderer, os.X_OK):
        raise ValueError(f"renderer is not executable: {renderer}")
    contract = read_json(render_contract.resolve(strict=True))
    artifacts = contract.get("artifacts") or {}
    if not {"storm", "cvac", "dcn"}.issubset(artifacts):
        raise ValueError("render contract must bind STORM, CVAC and DCN")
    checkpoint_hashes: dict[str, str] = {}
    for name, artifact in artifacts.items():
        path = Path(str(artifact.get("path") or "")).resolve(strict=True)
        expected = str(artifact.get("sha256") or "")
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"render artifact hash mismatch: {name}")
        checkpoint_hashes[name] = expected
    jobs = list(iter_jsonl(jobs_path.resolve(strict=True)))
    if not jobs:
        raise ValueError("render manifest is empty")
    slots = [(gpu, worker) for gpu in gpus for worker in range(workers_per_gpu)]
    locality_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        key = (
            str(job.get("scene_name") or ""),
            str(job.get("window_id") or job.get("context_id") or ""),
        )
        locality_groups.setdefault(key, []).append(job)
    shard_count = min(len(locality_groups), len(slots) * shards_per_worker)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    ordered_groups = sorted(
        locality_groups.values(),
        key=lambda values: (-len(values), _job_identity(values[0])),
    )
    for group in ordered_groups:
        target = min(range(len(shards)), key=lambda index: (len(shards[index]), index))
        shards[target].extend(sorted(group, key=_job_identity))
    output_root.mkdir(parents=True, exist_ok=True)
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    async def execute(index: int, gpu: int, worker: int) -> dict[str, Any]:
        shard_jobs = shards[index]
        if not shard_jobs:
            return {
                "slot": index,
                "gpu": gpu,
                "worker": worker,
                "job_count": 0,
                "returncode": 0,
            }
        shard = shard_root / f"shard{index:04d}"
        shard.mkdir(parents=True, exist_ok=True)
        expected_rows = {_job_identity(row): row for row in shard_jobs}
        if len(expected_rows) != len(shard_jobs):
            raise ValueError("render shard contains duplicate job identities")

        def validated(result_path: Path) -> None:
            results: dict[str, dict[str, Any]] = {}
            for result in iter_jsonl(result_path):
                identity = _job_identity(result)
                if identity in results:
                    raise RuntimeError("result membership is duplicated")
                results[identity] = result
            if set(results) != set(expected_rows):
                raise RuntimeError("result membership differs")
            for identity, result in results.items():
                job = expected_rows[identity]
                if result.get("job_sha256") != job.get("job_sha256"):
                    raise RuntimeError(f"result is not bound to its job: {identity}")
                if result.get("checkpoint_sha256") != checkpoint_hashes:
                    raise RuntimeError(f"result checkpoint differs: {identity}")
                if job.get("job_kind") == "sixcam_baseline_context":
                    unsigned = dict(result)
                    claimed = str(unsigned.pop("result_sha256", ""))
                    status = result.get("status")
                    if status not in {"complete", "failed"}:
                        raise RuntimeError(
                            f"baseline result status differs: {identity}"
                        )
                    if (
                        claimed != canonical_sha256(unsigned)
                        or result.get("job_kind") != job.get("job_kind")
                        or result.get("context_id") != job.get("context_id")
                        or result.get("scene_name") != job.get("scene_name")
                        or result.get("split") != job.get("split")
                    ):
                        raise RuntimeError(
                            f"baseline result contract differs: {identity}"
                        )
                    if status == "failed":
                        error = result.get("error") or {}
                        if result.get("views") or not str(error.get("type") or ""):
                            raise RuntimeError(
                                f"baseline failure receipt differs: {identity}"
                            )
                        continue
                    expected_views = {
                        (int(row["frame_index"]), str(row["camera_id"])): row
                        for row in job.get("requested_views") or []
                    }
                    observed_views = {
                        (int(row.get("frame_index", -1)), str(row.get("camera_id") or "")): row
                        for row in result.get("views") or []
                    }
                    if (
                        set(observed_views) != set(expected_views)
                        or len(observed_views) != len(result.get("views") or [])
                    ):
                        raise RuntimeError(f"baseline result contract differs: {identity}")
                    for key, view in observed_views.items():
                        if (
                            view.get("exposure") != expected_views[key].get("exposure")
                            or set(view.get("paths") or {}) != {"gt", "target"}
                            or set(view.get("content_sha256") or {}) != {"gt", "target"}
                            or any(
                                len(str(value or "")) != 64
                                for value in (view.get("content_sha256") or {}).values()
                            )
                        ):
                            raise RuntimeError(
                                f"baseline view result contract differs: {identity}:{key}"
                            )
                elif result.get("status") == "failed":
                    unsigned = dict(result)
                    claimed = str(unsigned.pop("result_sha256", ""))
                    error = result.get("error") or {}
                    if (
                        claimed != canonical_sha256(unsigned)
                        or not str(error.get("type") or "")
                        or result.get("scene_name") != job.get("scene_name")
                        or int(result.get("frame_index", -1))
                        != int(job.get("frame_index", -2))
                        or str(result.get("camera_id") or "")
                        != str(job.get("camera_id") or "")
                        or list(result.get("selected_obj_ids") or [])
                        != list(job.get("selected_obj_ids") or [])
                    ):
                        raise RuntimeError(
                            f"failed render receipt is not bound to its job: {identity}"
                        )
                elif result.get("status") != "complete" or (
                    list(result.get("selected_obj_ids") or [])
                    != list(job["selected_obj_ids"])
                    or str(result.get("camera_id")) != str(job["camera_id"])
                    or int(result.get("frame_index", -1)) != int(job["frame_index"])
                ):
                    raise RuntimeError(f"result is not bound to its job: {identity}")

        attempts = sorted(path for path in shard.glob("attempt*") if path.is_dir())
        for attempt in reversed(attempts):
            result_path = attempt / "results.jsonl"
            if not result_path.is_file():
                continue
            try:
                validated(result_path)
            except Exception:
                continue
            return {
                "slot": index,
                "gpu": gpu,
                "worker": worker,
                "job_count": len(shard_jobs),
                "returncode": 0,
                "resumed": True,
                "results": str(result_path.resolve()),
            }
        attempt_index = len(attempts)
        root = shard / f"attempt{attempt_index:04d}"
        while root.exists():
            attempt_index += 1
            root = shard / f"attempt{attempt_index:04d}"
        root.mkdir(parents=True, exist_ok=False)
        manifest = root / "jobs.jsonl"
        result_path = root / "results.jsonl"
        log_path = root / "renderer.log"
        atomic_jsonl(manifest, shard_jobs)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["DRIVEHARM_PHYSICAL_GPU"] = str(gpu)
        environment["PATH"] = os.pathsep.join(
            [str(renderer.parent), environment.get("PATH", "")]
        )
        process = await asyncio.create_subprocess_exec(
            str(renderer),
            *renderer_args,
            "--jobs",
            str(manifest),
            "--output-root",
            str(root / "outputs"),
            "--results",
            str(result_path),
            "--gpu",
            "0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        try:
            stdout, _ = await process.communicate()
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()
            raise
        log_path.write_bytes(stdout or b"")
        if process.returncode != 0:
            raise RuntimeError(
                f"renderer slot {index} exited with {process.returncode}"
            )
        validated(result_path)
        return {
            "slot": index,
            "gpu": gpu,
            "worker": worker,
            "job_count": len(shard_jobs),
            "returncode": process.returncode,
            "resumed": False,
            "results": str(result_path.resolve()),
        }

    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(len(shards)):
        queue.put_nowait(index)

    async def slot_worker(gpu: int, worker: int) -> list[dict[str, Any]]:
        completed = []
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return completed
            try:
                completed.append(await execute(index, gpu, worker))
            finally:
                queue.task_done()

    grouped = await asyncio.gather(*(slot_worker(gpu, worker) for gpu, worker in slots))
    executions = sorted(
        (row for group in grouped for row in group), key=lambda row: row["slot"]
    )
    result_rows: list[dict[str, Any]] = []
    for execution in executions:
        raw = execution.get("results")
        if raw:
            result_rows.extend(iter_jsonl(Path(str(raw))))
    result_rows.sort(key=_job_identity)
    if len(result_rows) != len(jobs):
        raise RuntimeError("combined render result count differs")
    combined = output_root / "results.jsonl"
    atomic_jsonl(combined, result_rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "job_count": len(jobs),
        "successful_job_count": sum(
            row.get("status") == "complete" for row in result_rows
        ),
        "failed_job_count": sum(row.get("status") == "failed" for row in result_rows),
        "gpu_ids": list(gpus),
        "workers_per_gpu": workers_per_gpu,
        "checkpoint_sha256": checkpoint_hashes,
        "slot_count": len(slots),
        "shard_count": len(shards),
        "shards_per_worker": shards_per_worker,
        "shard_policy": "scene_window_locality_balanced_queue",
        "resumed_shard_count": sum(bool(row.get("resumed")) for row in executions),
        "executions": executions,
        "results": str(combined.resolve()),
        "results_sha256": canonical_sha256(result_rows),
    }
    atomic_json(output_root / "summary.json", summary)
    return summary
