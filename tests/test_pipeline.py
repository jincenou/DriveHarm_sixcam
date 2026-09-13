from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from driveharm.audit import audit_release
from driveharm.cli import _accepted_jobs, _asset_review_manifest
from driveharm.compose import compose_results, edit_overlap_pass, geometry_pass
from driveharm.contracts import atomic_json, atomic_jsonl, canonical_sha256, sha256_file
from driveharm.planning import _window_selections, build_plan
from driveharm.release import publish_release, quarantine_triplets
import driveharm.render as render_module
import driveharm.review as review_module
import driveharm.sixcam_production as production_module
from driveharm.storm_adapter import _legacy_index


SIZE = (512, 288)


def save_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8), mode="RGB").save(path)


def save_mask(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8), mode="L").save(path)


class PipelineTests(unittest.TestCase):
    def test_legacy_render_lookup_is_context_specific(self) -> None:
        opportunities = []
        for index, context in enumerate(("scene-0001__w000010__aaa", "scene-0001__w000010__bbb")):
            opportunities.append(
                {
                    "job_index": index,
                    "multi_case_id": context,
                    "scene_name": "scene-0001",
                    "obj_id": "shared-object",
                    "target_usable_by_frame_camera": {"000013:c0": True},
                }
            )
        index = _legacy_index({"jobs": opportunities})
        self.assertEqual(len(index), 2)
        self.assertEqual(
            index[("scene-0001__w000010__aaa", "shared-object", 13, "0")][
                "job_index"
            ],
            0,
        )
        self.assertEqual(
            index[("scene-0001__w000010__bbb", "shared-object", 13, "0")][
                "job_index"
            ],
            1,
        )

    def test_train_window_subsets_are_instance_safe_and_capped(self) -> None:
        def member(obj_id: str, instance: str) -> dict:
            return {"assets": [{"obj_id": obj_id, "instance_token": instance}]}

        members = {
            "a1": member("a1", "actor-a"),
            "a2": member("a2", "actor-a"),
            "b1": member("b1", "actor-b"),
            "c1": member("c1", "actor-c"),
        }
        first = _window_selections(members, 6, 20260818, "scene:w000001")
        second = _window_selections(members, 6, 20260818, "scene:w000001")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue({("a1",), ("a2",), ("b1",), ("c1",)}.issubset(first))
        self.assertFalse(any({"a1", "a2"}.issubset(selection) for selection in first))

    def test_train_exact_capacity_and_identity_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "assets/car").mkdir(parents=True)
            ply = root / "assets/car/car.ply"
            ply.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
            asset_sha = sha256_file(ply)
            exact = root / "exact_assets.json"
            atomic_json(
                exact,
                {
                    "assets": [
                        {
                            "global_uid": "nuscenes_car_scene-0001__token-1",
                            "obj_id": "scene-0001__token-1",
                            "instance_token": "token-1",
                            "category": "car",
                            "asset_path": "assets/car/car.ply",
                            "size_xyz": [1, 1, 1],
                            "canonical": {
                                "canonical_asset_sha256": asset_sha,
                                "front_axis": "+X",
                            },
                        }
                    ]
                },
            )
            post = root / "post.json"
            atomic_json(
                post,
                {
                    "status": "complete",
                    "manifest": exact.name,
                    "manifest_sha256": sha256_file(exact),
                },
            )
            sheet = root / "sheet.png"
            save_rgb(sheet, np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8))
            identity = root / "identity.jsonl"
            atomic_jsonl(
                identity,
                [
                    {
                        "obj_id": "scene-0001__token-1",
                        "instance_token": "token-1",
                        "category": "car",
                        "canonical_asset_sha256": asset_sha,
                        "review_sheet": str(sheet),
                        "review_sheet_sha256": sha256_file(sheet),
                    }
                ],
            )
            capacity = root / "exact_asset_windows.jsonl"
            atomic_jsonl(
                capacity,
                [
                    {
                        "scene_name": "scene-0001",
                        "context_start": 10,
                        "context_frames": [10, 15, 20, 25],
                        "target_frames": [13, 17],
                        "quality_tier": "high",
                        "candidates": [
                            {
                                "obj_id": "scene-0001__token-1",
                                "instance_token": "token-1",
                                "official_size_wlh_m": [1.8, 4.2, 1.6],
                                "projected_area_native_pixels": 300,
                                "visibility_token": 4,
                                "visible_target_frame_camera_keys": [
                                    "000013:c0",
                                    "000017:c3",
                                ],
                            }
                        ],
                    }
                ],
            )
            summary = build_plan(post, capacity, root / "plan", identity)
            self.assertEqual(summary["job_count"], 2)
            rows = [
                json.loads(line)
                for line in Path(summary["jobs"]).read_text().splitlines()
            ]
            self.assertEqual({row["camera_id"] for row in rows}, {"0", "3"})
            self.assertTrue(
                all(row["official_dimensions_m"] == [4.2, 1.8, 1.6] for row in rows)
            )

    def test_00_multi_gpu_batch_dispatch_binds_results(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            jobs = []
            for index in range(6):
                job = {
                    "sample_id": f"sample-{index}",
                    "job_sha256": f"hash-{index}",
                    "selected_obj_ids": [f"obj-{index}"],
                    "camera_id": str(index % 6),
                    "frame_index": index,
                }
                jobs.append(job)
            jobs_path = root / "jobs.jsonl"
            atomic_jsonl(jobs_path, jobs)
            renderer = root / "renderer.py"
            renderer.write_text("executable marker\n", encoding="utf-8")
            renderer.chmod(0o755)
            checkpoints = {}
            checkpoint_hashes = {}
            for name in ("storm", "cvac", "dcn"):
                path = root / f"{name}.ckpt"
                path.write_text(name, encoding="utf-8")
                checkpoint_hashes[name] = sha256_file(path)
                checkpoints[name] = {
                    "path": str(path),
                    "sha256": checkpoint_hashes[name],
                }
            render_contract = root / "render_contract.json"
            atomic_json(render_contract, {"artifacts": checkpoints})
            calls = {"count": 0, "paths": []}

            class FakeProcess:
                returncode = 0

                async def communicate(self):
                    await asyncio.sleep(0.01)
                    return b"ok", None

            async def create_process(*args, **kwargs):
                calls["count"] += 1
                calls["paths"].append(kwargs["env"]["PATH"])
                arguments = list(args)
                manifest = Path(arguments[arguments.index("--jobs") + 1])
                results = Path(arguments[arguments.index("--results") + 1])
                rows = [json.loads(line) for line in manifest.read_text().splitlines()]
                result_rows = []
                for row in rows:
                    result = {
                        **{
                            key: row[key]
                            for key in (
                                "sample_id",
                                "job_sha256",
                                "selected_obj_ids",
                                "camera_id",
                                "frame_index",
                            )
                        },
                        "status": "complete",
                        "checkpoint_sha256": checkpoint_hashes,
                    }
                    if row["sample_id"] == "sample-5":
                        result["status"] = "failed"
                        result["error"] = {
                            "type": "BoundedFixtureFailure",
                            "message": "quarantine this one job",
                        }
                        result["result_sha256"] = canonical_sha256(result)
                    result_rows.append(result)
                atomic_jsonl(
                    results,
                    result_rows,
                )
                return FakeProcess()

            original = render_module.asyncio.create_subprocess_exec
            render_module.asyncio.create_subprocess_exec = create_process
            try:
                summary = asyncio.run(
                    render_module.render_shards(
                        jobs_path,
                        root / "render",
                        renderer,
                        render_contract,
                        gpus=(0, 1),
                    )
                )
            finally:
                render_module.asyncio.create_subprocess_exec = original
            self.assertEqual(summary["job_count"], 6)
            self.assertEqual(summary["gpu_ids"], [0, 1])
            self.assertEqual(summary["successful_job_count"], 5)
            self.assertEqual(summary["failed_job_count"], 1)
            self.assertTrue(
                all(
                    value.split(os.pathsep, 1)[0] == str(renderer.parent)
                    for value in calls["paths"]
                )
            )
            self.assertEqual(sum(row["job_count"] for row in summary["executions"]), 6)
            resumed = asyncio.run(
                render_module.render_shards(
                    jobs_path,
                    root / "render",
                    renderer,
                    render_contract,
                    gpus=(0, 1),
                )
            )
            self.assertEqual(calls["count"], 1)
            self.assertEqual(resumed["resumed_shard_count"], 1)

    def test_baseline_context_dispatch_validates_requested_views(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoints = {}
            checkpoint_hashes = {}
            for name in ("storm", "cvac", "dcn"):
                path = root / f"{name}.ckpt"
                path.write_text(name, encoding="utf-8")
                checkpoint_hashes[name] = sha256_file(path)
                checkpoints[name] = {
                    "path": str(path),
                    "sha256": checkpoint_hashes[name],
                }
            contract = root / "contract.json"
            atomic_json(contract, {"artifacts": checkpoints})
            exposure = {
                "camera_name": "CAM_FRONT",
                "camera_exposure_timestamp_us": 100,
                "sample_data_token": "sample-data",
                "sample_token": "sample",
            }
            job = {
                "job_kind": "sixcam_baseline_context",
                "job_id": "baseline-one",
                "job_sha256": "job-hash",
                "split": "train",
                "context_id": "scene-0001__w000001__aaaaaaaaaaaa",
                "scene_name": "scene-0001",
                "checkpoint_sha256": checkpoint_hashes,
                "requested_views": [
                    {"frame_index": 4, "camera_id": "0", "exposure": exposure}
                ],
            }
            jobs = root / "jobs.jsonl"
            atomic_jsonl(jobs, [job])
            renderer = root / "renderer.py"
            renderer.write_text("executable marker\n", encoding="utf-8")
            renderer.chmod(0o755)

            class FakeProcess:
                returncode = 0

                async def communicate(self):
                    return b"ok", None

            async def create_process(*arguments, **_kwargs):
                values = list(arguments)
                result_path = Path(values[values.index("--results") + 1])
                view = {
                    "frame_index": 4,
                    "camera_id": "0",
                    "exposure": exposure,
                    "paths": {"gt": "/gt.png", "target": "/target.png"},
                    "content_sha256": {"gt": "a" * 64, "target": "b" * 64},
                }
                result = {
                    "status": "complete",
                    "job_kind": job["job_kind"],
                    "job_id": job["job_id"],
                    "job_sha256": job["job_sha256"],
                    "split": job["split"],
                    "context_id": job["context_id"],
                    "scene_name": job["scene_name"],
                    "checkpoint_sha256": checkpoint_hashes,
                    "views": [view],
                }
                result["result_sha256"] = canonical_sha256(result)
                atomic_jsonl(result_path, [result])
                return FakeProcess()

            original = render_module.asyncio.create_subprocess_exec
            render_module.asyncio.create_subprocess_exec = create_process
            try:
                summary = asyncio.run(
                    render_module.render_shards(
                        jobs,
                        root / "render",
                        renderer,
                        contract,
                        gpus=(0, 1),
                    )
                )
            finally:
                render_module.asyncio.create_subprocess_exec = original
            self.assertEqual(summary["job_count"], 1)
            self.assertEqual(summary["successful_job_count"], 1)
            self.assertEqual(summary["failed_job_count"], 0)
            self.assertEqual(summary["checkpoint_sha256"], checkpoint_hashes)

    def test_baseline_context_dispatch_tolerates_signed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts = {}
            hashes = {}
            for name in ("storm", "cvac", "dcn"):
                path = root / name
                path.write_text(name, encoding="utf-8")
                hashes[name] = sha256_file(path)
                artifacts[name] = {"path": str(path), "sha256": hashes[name]}
            contract = root / "contract.json"
            atomic_json(contract, {"artifacts": artifacts})
            job = {
                "job_kind": "sixcam_baseline_context",
                "job_id": "baseline-failed",
                "job_sha256": "failed-job-hash",
                "split": "train",
                "context_id": "scene-0001__w000002__bbbbbbbbbbbb",
                "scene_name": "scene-0001",
                "checkpoint_sha256": hashes,
                "requested_views": [
                    {"frame_index": 5, "camera_id": "0", "exposure": {}}
                ],
            }
            jobs = root / "jobs.jsonl"
            atomic_jsonl(jobs, [job])
            renderer = root / "renderer"
            renderer.write_text("marker\n", encoding="utf-8")
            renderer.chmod(0o755)

            class FakeProcess:
                returncode = 0

                async def communicate(self):
                    return b"failed but continued", None

            async def create_process(*arguments, **_kwargs):
                values = list(arguments)
                result_path = Path(values[values.index("--results") + 1])
                result = {
                    "schema_version": 1,
                    "status": "failed",
                    "job_kind": job["job_kind"],
                    "job_id": job["job_id"],
                    "job_sha256": job["job_sha256"],
                    "split": job["split"],
                    "context_id": job["context_id"],
                    "scene_name": job["scene_name"],
                    "checkpoint_sha256": hashes,
                    "views": [],
                    "error": {"type": "ValueError", "message": "bad context"},
                }
                result["result_sha256"] = canonical_sha256(result)
                atomic_jsonl(result_path, [result])
                return FakeProcess()

            original = render_module.asyncio.create_subprocess_exec
            render_module.asyncio.create_subprocess_exec = create_process
            try:
                summary = asyncio.run(
                    render_module.render_shards(
                        jobs, root / "render", renderer, contract, gpus=(0,)
                    )
                )
            finally:
                render_module.asyncio.create_subprocess_exec = original
            self.assertEqual(summary["successful_job_count"], 0)
            self.assertEqual(summary["failed_job_count"], 1)

    def test_strict_production_controller_uses_requested_four_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            indices = {}
            profiles = {}
            for split in ("train", "val"):
                index = root / f"{split}-index"
                index.mkdir()
                atomic_json(
                    index / "summary.json",
                    {
                        "status": "complete",
                        "split": split,
                        "content_verification": {"performed": True},
                    },
                )
                atomic_jsonl(index / "baseline_jobs.jsonl", [])
                atomic_jsonl(index / "visible_backfill_jobs.jsonl", [])
                indices[split] = index
                profile = root / f"{split}-profile.json"
                atomic_json(profile, {})
                profiles[split] = profile
            executable = root / "python"
            executable.write_text("marker\n", encoding="utf-8")
            executable.chmod(0o755)
            contract = root / "contract.json"
            atomic_json(contract, {})
            calls = []

            async def fake_render(jobs, output, *_args, **kwargs):
                output.mkdir(parents=True, exist_ok=True)
                results = output / "results.jsonl"
                atomic_jsonl(results, [])
                calls.append(
                    (
                        jobs.name,
                        tuple(kwargs["gpus"]),
                        kwargs["shards_per_worker"],
                    )
                )
                return {"status": "complete", "results": str(results)}

            def fake_compose(_results, output):
                output.mkdir(parents=True, exist_ok=True)
                records = output / "records.jsonl"
                atomic_jsonl(records, [])
                return {"status": "complete", "records": str(records)}

            def fake_publish(**_kwargs):
                return {"status": "complete", "group_count": 1}

            def fake_audit(*_args, **_kwargs):
                return {"status": "pass"}

            originals = (
                production_module.render_shards,
                production_module.compose_results,
                production_module.publish_strict_sixcam_release,
                production_module.audit_strict_sixcam_release,
            )
            production_module.render_shards = fake_render
            production_module.compose_results = fake_compose
            production_module.publish_strict_sixcam_release = fake_publish
            production_module.audit_strict_sixcam_release = fake_audit
            try:
                asyncio.run(
                    production_module.run_strict_sixcam_production(
                        train_index=indices["train"],
                        val_index=indices["val"],
                        train_profile=profiles["train"],
                        val_profile=profiles["val"],
                        baseline_python=executable,
                        visible_python=executable,
                        render_contract=contract,
                        work_root=root / "work",
                        destination=root / "destination",
                        receipt_root=root / "receipt",
                        gpus=(0, 1, 2, 3),
                    )
                )
            finally:
                (
                    production_module.render_shards,
                    production_module.compose_results,
                    production_module.publish_strict_sixcam_release,
                    production_module.audit_strict_sixcam_release,
                ) = originals
            self.assertEqual(len(calls), 4)
            self.assertTrue(
                all(gpus == (0, 1, 2, 3) for _name, gpus, _shards in calls)
            )
            self.assertEqual(
                [shards for name, _gpus, shards in calls if name == "baseline_jobs.jsonl"],
                [4, 4],
            )
            self.assertEqual(
                [
                    shards
                    for name, _gpus, shards in calls
                    if name == "visible_backfill_jobs.jsonl"
                ],
                [1, 1],
            )
            self.assertEqual(
                json.loads((root / "work/production_state.json").read_text())[
                    "status"
                ],
                "complete",
            )

    def test_async_json_schema_review_is_concurrent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image_path = root / "view.png"
            save_rgb(image_path, np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8))
            manifest = root / "review.jsonl"
            atomic_jsonl(
                manifest,
                [
                    {
                        "sample_id": f"sample-{index}",
                        "review_images": {"input": str(image_path)},
                    }
                    for index in range(8)
                ],
            )
            state = {"active": 0, "maximum": 0, "formats": []}

            class FakeCompletions:
                async def create(self, **kwargs):
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                    state["formats"].append(kwargs["response_format"])
                    await asyncio.sleep(0.01)
                    state["active"] -= 1
                    decision = {
                        "decision": "accept",
                        "same_identity": True,
                        "broken_or_doubled_asset": False,
                        "orientation_valid": True,
                        "scale_valid": True,
                        "grounding_valid": True,
                        "severe_occlusion_error": False,
                        "confidence": 0.9,
                        "reason": "ok",
                    }
                    message = type("Message", (), {"content": json.dumps(decision)})()
                    choice = type("Choice", (), {"message": message})()
                    return type("Response", (), {"choices": [choice]})()

            class FakeClient:
                def __init__(self, **kwargs):
                    self.chat = type("Chat", (), {"completions": FakeCompletions()})()

                async def close(self):
                    return None

            original = review_module.AsyncOpenAI
            review_module.AsyncOpenAI = FakeClient
            try:
                summary = asyncio.run(
                    review_module.review_manifest(
                        manifest,
                        root / "reviews",
                        "http://localhost/v1",
                        "local",
                        "model",
                        4,
                    )
                )
            finally:
                review_module.AsyncOpenAI = original
            self.assertEqual(summary["accepted_count"], 8)
            self.assertGreaterEqual(state["maximum"], 2)
            self.assertTrue(
                all(value["type"] == "json_schema" for value in state["formats"])
            )
            reused = asyncio.run(
                review_module.review_manifest(
                    manifest,
                    root / "reviews",
                    "http://localhost/v1",
                    "local",
                    "model",
                    4,
                )
            )
            self.assertEqual(reused["reused_count"], 8)
            self.assertEqual(reused["requested_count"], 0)

    def test_any_camera_plan_and_asset_hash_binding(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ply = root / "asset.ply"
            ply.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
            ply2 = root / "asset2.ply"
            ply2.write_bytes(b"ply\nformat ascii 1.0\ncomment second\nend_header\n")
            save_rgb(
                root / "unused.png", np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
            )
            save_rgb(
                root / "unused2.png", np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
            )
            manifest = root / "assets.jsonl"
            atomic_jsonl(
                manifest,
                [
                    {
                        "asset_id": "asset-0001",
                        "obj_id": "obj-1",
                        "instance_token": "token-1",
                        "category": "car",
                        "ply_path": str(ply),
                        "ply_sha256": sha256_file(ply),
                        "dimensions_m": [4.2, 1.8, 1.6],
                        "forward_axis": "+X",
                    },
                    {
                        "asset_id": "asset-0002",
                        "obj_id": "obj-2",
                        "instance_token": "token-2",
                        "category": "car",
                        "ply_path": str(ply2),
                        "ply_sha256": sha256_file(ply2),
                        "dimensions_m": [4.0, 1.7, 1.5],
                        "forward_axis": "+X",
                    },
                ],
            )
            post = root / "post.json"
            atomic_json(
                post,
                {
                    "status": "complete",
                    "manifest": str(manifest),
                    "manifest_sha256": sha256_file(manifest),
                },
            )
            observations = root / "observations.jsonl"
            atomic_jsonl(
                observations,
                [
                    {
                        "scene_name": "scene-0001",
                        "window_id": "w000010",
                        "frame_index": 12,
                        "obj_id": "obj-1",
                        "instance_token": "token-1",
                        "area_pixels": 300,
                        "visibility_level": 4,
                        "visible_cameras": ["c3", "0"],
                        "review_images": {"reference": str(root / "unused.png")},
                    },
                    {
                        "scene_name": "scene-0001",
                        "window_id": "w000010",
                        "frame_index": 12,
                        "obj_id": "obj-2",
                        "instance_token": "token-2",
                        "area_pixels": 250,
                        "visibility_level": 4,
                        "visible_cameras": ["c3"],
                        "review_images": {"reference": str(root / "unused2.png")},
                    },
                ],
            )
            summary = build_plan(post, observations, root / "plan")
            self.assertEqual(summary["job_count"], 4)
            self.assertEqual(summary["combination_job_count"], 1)
            self.assertEqual(summary["camera_counts"], {"0": 1, "3": 3})
            self.assertEqual(summary["camera_policy"], "any_visible_camera")
            asset_reviews = root / "asset_reviews.jsonl"
            _asset_review_manifest(Path(summary["jobs"]), asset_reviews)
            self.assertEqual(len(asset_reviews.read_text().splitlines()), 2)
            decisions = root / "decisions.jsonl"
            atomic_jsonl(
                decisions,
                [
                    {"sample_id": "asset-0001", "hard_failure": False},
                    {"sample_id": "asset-0002", "hard_failure": True},
                ],
            )
            reviewed_jobs = root / "reviewed_jobs.jsonl"
            _accepted_jobs(Path(summary["jobs"]), decisions, reviewed_jobs)
            self.assertEqual(len(reviewed_jobs.read_text().splitlines()), 2)

    def test_compose_occlusion_audit_release_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = root / "sources"
            sources.mkdir()
            grey = np.full((SIZE[1], SIZE[0], 3), 100, dtype=np.uint8)
            removed = grey.copy()
            asset = np.zeros_like(grey)
            alpha = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
            alpha[80:180, 150:250] = 255
            removed[80:180, 150:250] = 0
            asset[80:180, 150:250, 0] = 255
            foreground = np.zeros_like(alpha)
            foreground[80:100, 150:250] = 255
            exact = foreground.copy()
            paths = {
                "real_gt": sources / "gt.png",
                "storm_baseline": sources / "target.png",
                "actor_removed": sources / "removed.png",
                "actor_removed_mask": sources / "removed_mask.png",
                "rgb": sources / "asset.png",
                "alpha": sources / "alpha.png",
                "foreground": sources / "foreground.png",
                "exact": sources / "exact.png",
            }
            save_rgb(paths["real_gt"], grey)
            save_rgb(paths["storm_baseline"], grey)
            save_rgb(paths["actor_removed"], removed)
            save_mask(paths["actor_removed_mask"], alpha)
            save_rgb(paths["rgb"], asset)
            save_mask(paths["alpha"], alpha)
            save_mask(paths["foreground"], foreground)
            save_mask(paths["exact"], exact)
            quality = {
                "target_actor_pixels": 10000,
                "complete_projected_asset_pixels": 10000,
                "silhouette_iou": 0.9,
                "center_error_px": 0,
                "width_ratio": 1,
                "height_ratio": 1,
                "bottom_error_px": 0,
                "orientation_error_deg": 0,
                "ground_lock_pass": True,
                "catastrophic_asset_safety_pass": True,
                "broken_or_doubled_asset": False,
                "verified_foreground_occlusion_applied": True,
                "degenerate_rectangle_mask": False,
            }
            cleanup = {"observation_technical_usable": True, "quality_gate_pass": True}
            cleanup["decision_sha256"] = canonical_sha256(cleanup)
            result = {
                "status": "complete",
                "sample_id": "scene-0001__w000010__asset__f012_c3",
                "scene_name": "scene-0001",
                "frame_index": 12,
                "camera_id": "3",
                "render_domain": "storm",
                "camera_exposure": {
                    "sample_data_timestamp_us": 123456,
                    "camera_exposure_timestamp_us": 123456,
                    "storm_normalized_time_seconds": 1.25,
                },
                "original_actor_cleanup_receipt": cleanup,
                "quality": {
                    "target_storm_vs_sensor_before_insertion": {
                        "quality_gate_pass": True
                    }
                },
                "paths": {
                    key: str(paths[key])
                    for key in (
                        "real_gt",
                        "storm_baseline",
                        "actor_removed",
                        "actor_removed_mask",
                    )
                },
                "source_sha256": {
                    key: sha256_file(paths[key])
                    for key in (
                        "real_gt",
                        "storm_baseline",
                        "actor_removed",
                        "actor_removed_mask",
                    )
                },
                "asset_layers": [
                    {
                        "obj_id": "obj-1",
                        "instance_token": "token-1",
                        "asset_sha256": "a" * 64,
                        "exact_identity_verified": True,
                        "rgb": str(paths["rgb"]),
                        "alpha": str(paths["alpha"]),
                        "official_dimensions_m": [4.2, 1.8, 1.6],
                        "forward_axis": "+X",
                        "official_dimensions_applied": True,
                        "canonical_orientation_applied": True,
                        "bottom_ground_locked": True,
                        "content_sha256": {
                            "rgb": sha256_file(paths["rgb"]),
                            "alpha": sha256_file(paths["alpha"]),
                            "foreground_occlusion_mask": sha256_file(
                                paths["foreground"]
                            ),
                            "target_exact_mask": sha256_file(paths["exact"]),
                        },
                        "median_depth_m": 10,
                        "quality": quality,
                        "foreground_occlusion_mask": str(paths["foreground"]),
                        "target_exact_mask": str(paths["exact"]),
                        "foreground_occlusion_receipt": {
                            "independent_foreground_verified": True,
                            "renderer_verified_occlusion_pixels": 2000,
                            "restored_asset_pixels": 0,
                            "official_instance_decisions": [
                                {
                                    "is_distinct_from_target": True,
                                    "official_annotation_bound": True,
                                    "strictly_nearer_than_target": True,
                                }
                            ],
                        },
                    }
                ],
            }
            results = root / "results.jsonl"
            atomic_jsonl(results, [result])
            for role in ("gt", "input", "target"):
                stale = root / "composed" / role / "stale.png"
                stale.parent.mkdir(parents=True, exist_ok=True)
                save_rgb(stale, grey)
            composed = compose_results(results, root / "composed")
            self.assertFalse((root / "composed/gt/stale.png").exists())
            candidate_rows = [
                json.loads(line)
                for line in (root / "composed/candidates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(composed["accepted_count"], 1, candidate_rows)
            with Image.open(
                root / "composed/input" / f"{result['sample_id']}.png"
            ) as image:
                output = np.asarray(image)
            self.assertTrue(np.all(output[85, 175] == 0))
            self.assertEqual(int(output[120, 175, 0]), 255)
            audited = audit_release(
                root / "composed", Path(composed["records"]), root / "audit"
            )
            self.assertEqual(audited["accepted_count"], 1)
            destination = root / "release"
            receipt = publish_release(
                [root / "composed", root / "composed"],
                [Path(audited["accepted_records"]), Path(audited["accepted_records"])],
                destination,
                root / "receipts",
                "copy",
            )
            self.assertEqual(receipt["triplet_count"], 1)
            self.assertEqual(receipt["duplicate_exclusion_count"], 1)
            candidate_ids = root / "candidate_ids.txt"
            candidate_ids.write_text(result["sample_id"] + "\n", encoding="utf-8")
            quarantined = quarantine_triplets(
                destination,
                candidate_ids,
                root / "quarantine",
                root / "quarantine.json",
            )
            self.assertEqual(quarantined["retained_count"], 0)

    def test_geometry_gate_rejects_clear_reversal(self) -> None:
        quality = {
            "target_actor_pixels": 2000,
            "complete_projected_asset_pixels": 2000,
            "silhouette_iou": 0.8,
            "center_error_px": 1,
            "width_ratio": 1,
            "height_ratio": 1,
            "bottom_error_px": 1,
            "orientation_error_deg": 170,
            "ground_lock_pass": True,
            "catastrophic_asset_safety_pass": True,
            "broken_or_doubled_asset": False,
        }
        self.assertFalse(geometry_pass(quality))

    def test_geometry_gate_preserves_strict_renderer_proof(self) -> None:
        quality = {
            "target_actor_pixels": 1680,
            "complete_projected_asset_pixels": 1680,
            "silhouette_iou": 0.74,
            "center_error_px": 2.5,
            "width_ratio": 1.01,
            "height_ratio": 0.76,
            "bottom_error_px": 4,
            "orientation_error_deg": 2,
            "ground_lock_pass": True,
            "catastrophic_asset_safety_pass": True,
            "broken_or_doubled_asset": False,
            "renderer_quality_gate_pass": True,
            "renderer_geometry_gate_pass": True,
        }
        self.assertTrue(geometry_pass(quality))
        quality["orientation_error_deg"] = 170
        # Contradictory clear-reversal evidence remains a hard failure even if
        # an upstream strict proof claims success.
        self.assertFalse(geometry_pass(quality))
        quality["orientation_error_deg"] = 2
        quality["renderer_geometry_gate_pass"] = False
        self.assertFalse(geometry_pass(quality))

    def test_train_two_percent_edit_overlap(self) -> None:
        first = np.zeros((20, 20), dtype=bool)
        second = np.zeros_like(first)
        first[:10, :10] = True
        second[10:20, 10:20] = True
        second[9, :3] = True
        self.assertFalse(edit_overlap_pass([first, second]))
        second[9, 2] = False
        self.assertTrue(edit_overlap_pass([first, second]))

    def test_static_depth_occlusion_receipt_is_accepted(self) -> None:
        from driveharm.compose import _verified_foreground

        receipt = {
            "schema_version": 1,
            "policy": "distinct_nearer_official_or_static_depth_v1",
            "target_instance_token": "a" * 32,
            "target_center_depth_m": 20.0,
            "absolute_depth_margin_m": 0.3,
            "relative_depth_margin": 0.02,
            "minimum_overlap_pixels": 16,
            "minimum_static_seed_pixels": 4,
            "image_shape_hw": [288, 512],
            "scene_depth_sha256": "a" * 64,
            "asset_depth_sha256": "b" * 64,
            "nearer_depth_support_sha256": "c" * 64,
            "official_instance_decisions": [],
            "static_region_decisions": [
                {
                    "evidence_kind": "unannotated_static_depth_seed_region",
                    "minimum_verified_seed_pixels_per_component": 4,
                    "components": [
                        {
                            "component_index": 1,
                            "component_pixels": 20,
                            "verified_nearer_seed_pixels": 6,
                            "accepted": True,
                        }
                    ],
                    "accepted_component_indices": [1],
                    "selected_pixels": 20,
                    "accepted": True,
                }
            ],
            "selected_occlusion_pixels": 20,
            "target_exact_is_diagnostic_only": True,
            "target_exact_overlap_pixels": 0,
            "quality_gate_pass": True,
            "independent_foreground_verified": True,
            "renderer_verified_occlusion_pixels": 20,
            "restored_asset_pixels": 0,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            foreground = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
            foreground[20:24, 20:25] = 255
            path = root / "foreground.png"
            save_mask(path, foreground)
            alpha = np.zeros((SIZE[1], SIZE[0]), dtype=np.float32)
            alpha[10:30, 10:30] = 1.0
            from driveharm.compose import _array_sha256

            receipt["full_asset_support_sha256"] = _array_sha256(
                (alpha > 0.02).astype(np.uint8)
            )
            receipt["selected_occlusion_mask_sha256"] = _array_sha256(
                (foreground > 0).astype(np.uint8)
            )
            receipt["decision_sha256"] = canonical_sha256(receipt)
            selected, summary = _verified_foreground(
                {
                    "foreground_occlusion_mask": str(path),
                    "foreground_occlusion_receipt": receipt,
                },
                alpha,
            )
            self.assertEqual(int(selected.sum()), 20)
            self.assertTrue(summary["applied"])

    def test_delivery_contracts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        python_files = sorted(root.glob("driveharm/*.py")) + sorted(
            root.glob("tests/*.py")
        )
        expected_python_files = {
            "driveharm/__init__.py",
            "driveharm/audit.py",
            "driveharm/cli.py",
            "driveharm/compose.py",
            "driveharm/contracts.py",
            "driveharm/planning.py",
            "driveharm/release.py",
            "driveharm/render.py",
            "driveharm/review.py",
            "driveharm/sixcam.py",
            "driveharm/sixcam_index.py",
            "driveharm/sixcam_pilot.py",
            "driveharm/sixcam_production.py",
            "driveharm/storm_adapter.py",
            "driveharm/storm_baseline.py",
            "tests/test_pipeline.py",
            "tests/test_sixcam.py",
            "tests/test_sixcam_index.py",
        }
        self.assertEqual(
            {str(path.relative_to(root)) for path in python_files},
            expected_python_files,
        )
        source = "\n".join(path.read_text(encoding="utf-8") for path in python_files)
        self.assertIn("AsyncOpenAI", source)
        self.assertIn("await client.chat.completions.create", source)
        self.assertIn('"type": "json_schema"', source)
        self.assertNotIn("re" + "quests", source.lower())
        self.assertNotIn("fall" + "back", source.lower())


if __name__ == "__main__":
    unittest.main()
