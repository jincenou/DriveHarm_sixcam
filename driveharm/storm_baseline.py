"""Efficient STORM/CVAC/DCN baseline export for strict six-camera jobs.

This module deliberately imports the frozen renderer at run time.  It keeps the
three models and the nuScenes adapter resident for every job in one shard, but
does not load SAM masks, remove an actor, rasterize an asset, or scan the large
trajectory plan once per context.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from .contracts import (
    IMAGE_SIZE,
    atomic_jsonl,
    canonical_sha256,
    iter_jsonl,
    sha256_file,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _regular_file(path: Path) -> Path:
    path = path.resolve(strict=True)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"renderer input is not a regular file: {path}")
    return path


def _checkpoint_contract(profile: dict[str, Any]) -> tuple[dict[str, str], dict[str, Path]]:
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name in ("storm", "cvac", "dcn"):
        path = _regular_file(Path(str(profile.get(f"{name}_checkpoint") or "")))
        paths[name] = path
        hashes[name] = sha256_file(path)
    return hashes, paths


def _atomic_rgb(path: Path, value: np.ndarray) -> None:
    if value.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) or value.dtype != np.uint8:
        raise ValueError(f"invalid baseline image array for {path}: {value.shape}/{value.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        Image.fromarray(value, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _FrozenRuntime:
    def __init__(self, profile: dict[str, Any], gpu: int) -> None:
        repo_root = Path(str(profile.get("repo_root") or "")).resolve(strict=True)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        render_pair = importlib.import_module("pipeline.render_pair")
        refine = importlib.import_module("pipeline.refine")
        geometry = importlib.import_module("pipeline.geometry")
        data_utils = importlib.import_module("storm.dataset.data_utils")
        cameras = importlib.import_module("pipeline.cameras")
        torch = importlib.import_module("torch")

        if gpu != 0:
            raise ValueError("the dispatcher must expose exactly one logical CUDA device")
        self.torch = torch
        self.render_pair = render_pair
        self.refine = refine
        self.data_utils = data_utils
        self.camera_ids = tuple(map(str, cameras.CAMERA_IDS))
        if self.camera_ids != ("1", "0", "2", "4", "5", "3"):
            raise RuntimeError("frozen STORM camera order changed")
        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("baseline export requires an exposed CUDA device")

        self.checkpoint_hashes, checkpoint_paths = _checkpoint_contract(profile)
        self.args = SimpleNamespace(
            data_root=Path(str(profile["data_root"])).resolve(strict=True),
            annotation_list=Path(str(profile["annotation_list"])).resolve(strict=True),
            raw_nuscenes_root=Path(str(profile["raw_nuscenes_root"])).resolve(strict=True),
            checkpoint=checkpoint_paths["storm"],
            input_size=[288, 512],
            num_cameras=6,
            timespan=2.0,
            model="STORM-L/16",
            model_num_cameras=6,
            num_motion_tokens=16,
            use_latest_gsplat=True,
            allow_unsafe_checkpoint=False,
        )
        torch.manual_seed(1)
        np.random.seed(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dataset = render_pair.make_nuscenes_dataset(self.args)
        self.scene_indices: dict[str, int] = {}
        for index, annotation in enumerate(self.dataset.annotations):
            scene = str(annotation.get("scene_name") or "")
            if not scene or scene in self.scene_indices:
                raise RuntimeError(f"STORM adapter repeats or omits a scene: {scene!r}")
            self.scene_indices[scene] = index
        self.model = geometry.construct_model(self.args, self.device)
        self.cvac, self.cvac_contract = refine.construct_cvac(
            checkpoint_paths["cvac"], self.device
        )
        self.dcn, self.dcn_contract = refine.construct_dcn(
            checkpoint_paths["dcn"], self.device
        )
        self.dcn_contract["checkpoint_sha256"] = self.checkpoint_hashes["dcn"]

    def render(self, job: dict[str, Any], output_root: Path) -> dict[str, Any]:
        torch = self.torch
        rp = self.render_pair
        started = time.perf_counter()
        job_id = str(job.get("job_id") or "")
        context_id = str(job.get("context_id") or "")
        scene = str(job.get("scene_name") or "")
        requested = job.get("requested_views") or []
        contract = (job.get("trajectory_plan") or {}).get("render_context") or {}
        if (
            job.get("job_kind") != "sixcam_baseline_context"
            or not job_id
            or not context_id
            or not scene
            or not requested
            or contract.get("case_id") != context_id
            or contract.get("scene_name") != scene
            or job.get("checkpoint_sha256") != self.checkpoint_hashes
        ):
            raise ValueError(f"invalid or mismatched baseline job: {job_id}")
        context_frames = list(map(int, contract.get("storm_context_frames") or []))
        target_frames = list(map(int, contract.get("storm_target_frames") or []))
        if len(context_frames) != 4 or len(target_frames) != 4:
            raise ValueError(f"baseline context is not a 4+4 STORM window: {job_id}")
        if tuple(map(str, contract.get("camera_ids") or ())) != self.camera_ids:
            raise ValueError(f"baseline context camera contract differs: {job_id}")
        dataset_index = self.scene_indices.get(scene)
        if dataset_index is None:
            raise ValueError(f"scene is absent from the STORM adapter: {scene}")

        sample = rp.exact_temporal_segment(
            self.dataset, dataset_index, context_frames, target_frames
        )
        input_dict, _target_dict = self.data_utils.prepare_inputs_and_targets(
            self.data_utils.to_batch_tensor(sample),
            device=self.device,
            v=6,
            timespan=2.0,
        )
        storm_annotation = self.dataset.annotations[dataset_index]
        rp.ensure_numeric_camera_keys(storm_annotation)
        context_metadata = rp.replace_context_with_native_six_camera(
            input_dict,
            storm_annotation,
            context_frames,
            self.args.raw_nuscenes_root,
            (288, 512),
            self.device,
        )
        raw_gt, target_metadata, native_target_time = rp.load_native_targets(
            self.args.raw_nuscenes_root,
            storm_annotation,
            target_frames,
            list(self.camera_ids),
            (288, 512),
            int(context_metadata["reference_front_exposure_timestamp_us"]),
            float(input_dict["timespan"]),
            self.device,
        )
        input_dict["target_time"] = native_target_time

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            gs_params = self.model.get_gs_params(input_dict)
            affine = rp.select_checkpoint_affine_camera_subset(
                gs_params,
                rendered_camera_count=6,
                affine_camera_indices=[0, 1, 2, 3, 4, 5],
            )
            baseline = rp.render_storm_target_time_chunks(
                self.model,
                gs_params,
                input_dict,
                num_cams=6,
                radius_clip=0.0,
                complete_rgb=True,
            )
            context_input = dict(input_dict)
            context_input["target_camtoworlds"] = input_dict["context_camtoworlds"]
            context_input["target_intrinsics"] = input_dict["context_intrinsics"]
            context_input["target_time"] = input_dict["context_time"]
            context_baseline = rp.render_storm_target_time_chunks(
                self.model,
                gs_params,
                context_input,
                num_cams=6,
                radius_clip=0.0,
                complete_rgb=True,
            )

        rgb_key = baseline["rgb_key"]
        baseline_01 = (baseline[rgb_key][0].float() * 0.5 + 0.5).clamp(0.0, 1.0)
        context_rgb_key = context_baseline["rgb_key"]
        context_01 = (
            context_baseline[context_rgb_key][0].float() * 0.5 + 0.5
        ).clamp(0.0, 1.0)
        context_sensor = (input_dict["context_image"].float() * 0.5 + 0.5).clamp(
            0.0, 1.0
        )
        baseline_cvac = self.refine.apply_cvac_per_frame(
            self.cvac,
            baseline_01.unsqueeze(0),
            baseline["rendered_alpha"][0].unsqueeze(0).float(),
            context_sensor,
        )
        context_prediction = self.refine.apply_cvac_per_frame(
            self.cvac,
            context_01.unsqueeze(0),
            context_baseline["rendered_alpha"][0].unsqueeze(0).float(),
            context_sensor,
        )
        corrected, _same, cvrc = self.refine.shared_cvrc_correction(
            baseline_cvac,
            baseline_cvac,
            baseline["rendered_depth"][0].float().unsqueeze(0),
            baseline["rendered_alpha"][0].float().unsqueeze(0),
            context_sensor.permute(0, 1, 2, 4, 5, 3).float(),
            context_prediction.float(),
            input_dict["context_camtoworlds"].float(),
            input_dict["context_intrinsics"].float(),
            input_dict["target_camtoworlds"].float(),
            input_dict["target_intrinsics"].float(),
            input_dict["context_time"].float(),
            input_dict["target_time"].float(),
            correction_weight=0.7,
            dcn=self.dcn,
        )
        target_rgb = (
            corrected[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0
        ).round().astype(np.uint8)
        metadata_by_key = {
            (int(row["target_frame"]), str(row["camera_id"])): row
            for row in target_metadata
        }
        requested_keys: set[tuple[int, str]] = set()
        views: list[dict[str, Any]] = []
        job_root = output_root / job_id
        for request in requested:
            frame = int(request.get("frame_index", -1))
            camera = str(request.get("camera_id") or "")
            key = (frame, camera)
            if key in requested_keys or frame not in target_frames or camera not in self.camera_ids:
                raise ValueError(f"invalid duplicate/out-of-window requested view: {job_id}:{key}")
            requested_keys.add(key)
            target_index = target_frames.index(frame)
            camera_index = self.camera_ids.index(camera)
            native = metadata_by_key[key]
            exposure = request.get("exposure") or {}
            if (
                int(exposure.get("camera_exposure_timestamp_us") or 0)
                != int(native["camera_exposure_timestamp_us"])
                or str(exposure.get("camera_name") or "")
                != str(native["camera_name"])
            ):
                raise ValueError(f"requested view exposure differs from native input: {job_id}:{key}")
            basename = f"f{frame:06d}_c{camera}"
            gt_path = job_root / f"{basename}__gt.png"
            target_path = job_root / f"{basename}__target.png"
            gt = raw_gt[target_index, camera_index]
            target = target_rgb[target_index, camera_index]
            _atomic_rgb(gt_path, gt)
            _atomic_rgb(target_path, target)
            metrics = rp.target_sensor_metrics(target, gt)
            metrics["quality_gate_pass"] = bool(
                metrics["psnr_db"] >= 20.0 and metrics["ssim"] >= 0.58
            )
            view = {
                "frame_index": frame,
                "camera_id": camera,
                "camera_name": native["camera_name"],
                "exposure": exposure,
                "paths": {
                    "gt": str(gt_path.resolve()),
                    "target": str(target_path.resolve()),
                },
                "content_sha256": {
                    "gt": sha256_file(gt_path),
                    "target": sha256_file(target_path),
                },
                "quality": metrics,
            }
            view["view_sha256"] = canonical_sha256(view)
            views.append(view)
        views.sort(key=lambda row: (row["frame_index"], self.camera_ids.index(row["camera_id"])))
        result = {
            "schema_version": 1,
            "status": "complete",
            "job_kind": "sixcam_baseline_context",
            "job_id": job_id,
            "job_sha256": job["job_sha256"],
            "split": job["split"],
            "context_id": context_id,
            "scene_name": scene,
            "checkpoint_sha256": self.checkpoint_hashes,
            "render_contract": {
                "model": "STORM-L/16",
                "native_size_wh": list(IMAGE_SIZE),
                "camera_order": list(self.camera_ids),
                "radius_clip": 0.0,
                "cvac": self.cvac_contract,
                "dcn": self.dcn_contract,
                "cvrc": cvrc,
                "checkpoint_affine": affine,
                "asset_pipeline_skipped": True,
            },
            "views": views,
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["result_sha256"] = canonical_sha256(result)
        return result


def render_jobs(args: argparse.Namespace) -> None:
    profile = _object(args.profile.resolve(strict=True))
    jobs = list(iter_jsonl(args.jobs.resolve(strict=True)))
    if not jobs or len({row.get("job_id") for row in jobs}) != len(jobs):
        raise ValueError("baseline jobs are empty or duplicated")
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime = _FrozenRuntime(profile, args.gpu)
    results: list[dict[str, Any]] = []
    progress_path = args.output_root / "progress.jsonl"
    with progress_path.open("a", encoding="utf-8") as progress:
        for ordinal, job in enumerate(jobs, 1):
            try:
                result = runtime.render(job, args.output_root)
            except Exception as error:
                message = str(error)
                fatal_markers = (
                    "out of memory",
                    "ninja is required",
                    "requires an exposed cuda device",
                    "cuda is not available",
                )
                if isinstance(error, MemoryError) or any(
                    marker in message.lower() for marker in fatal_markers
                ):
                    raise
                result = {
                    "schema_version": 1,
                    "status": "failed",
                    "job_kind": "sixcam_baseline_context",
                    "job_id": job["job_id"],
                    "job_sha256": job["job_sha256"],
                    "split": job["split"],
                    "context_id": job["context_id"],
                    "scene_name": job["scene_name"],
                    "checkpoint_sha256": runtime.checkpoint_hashes,
                    "views": [],
                    "error": {
                        "type": type(error).__name__,
                        "message": message[:2000],
                    },
                }
                result["result_sha256"] = canonical_sha256(result)
            results.append(result)
            progress.write(json.dumps(result, sort_keys=True) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            print(
                f"baseline progress {ordinal}/{len(jobs)} "
                f"{result['job_id']} {result['status']}",
                flush=True,
            )
    atomic_jsonl(args.results, results)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--jobs", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--results", type=Path, required=True)
    value.add_argument("--gpu", type=int, required=True)
    value.add_argument("--profile", type=Path, required=True)
    return value


def main() -> int:
    render_jobs(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
