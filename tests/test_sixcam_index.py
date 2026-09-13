from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from driveharm.contracts import atomic_json, atomic_jsonl, sha256_file
from driveharm.sixcam import CAMERA_RING
from driveharm.sixcam_index import build_strict_sixcam_index, iter_json_array


class StrictSixCameraIndexTests(unittest.TestCase):
    def test_streams_named_and_root_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            named = root / "named.json"
            root_array = root / "root.json"
            atomic_json(named, {"ignored": [0], "cases": [{"v": 1}, {"v": 2}]})
            root_array.write_text('[{"v":3}, {"v":4}]\n', encoding="utf-8")
            self.assertEqual(
                [row["v"] for row in iter_json_array(named, "cases")], [1, 2]
            )
            self.assertEqual(
                [row["v"] for row in iter_json_array(root_array)], [3, 4]
            )

    def test_val_schema_context_visibility_and_cross_token_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            for role in ("gt", "input", "target"):
                (source / role).mkdir(parents=True)

            asset_sha = "a" * 64
            manifest_sha = "b" * 64
            checkpoint_files = {}
            checkpoint_hashes = {}
            for name in ("storm", "cvac", "dcn"):
                path = root / f"{name}.ckpt"
                path.write_bytes(name.encode("utf-8"))
                checkpoint_files[name] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                checkpoint_hashes[name] = sha256_file(path)
            render_contract = root / "render_contract.json"
            atomic_json(render_contract, {"artifacts": checkpoint_files})

            contexts = (
                "scene-0003__w000010__111111111111",
                "scene-0003__w000010__222222222222",
            )
            records = []
            metadata = []
            for ordinal, context in enumerate(contexts):
                sample_id = f"{context}__cmb-aaaaaaaaaaaa__f013_c0"
                hashes = {}
                for role_index, role in enumerate(("gt", "input", "target")):
                    value = 30 + ordinal * 40 + role_index
                    image = np.full((288, 512, 3), value, dtype=np.uint8)
                    path = source / role / f"{sample_id}.png"
                    Image.fromarray(image, mode="RGB").save(path)
                    hashes[role] = sha256_file(path)
                # Final val authority rows deliberately omit scene/frame fields.
                records.append(
                    {
                        "sample_id": sample_id,
                        "camera_id": "0",
                        # Some final authority rows intentionally bind only the
                        # canonical UID.  Missing optional identity fields are
                        # not a disagreement with the frozen visibility record.
                        **(
                            {
                                "assets": [
                                    {
                                        "global_uid": "asset-one",
                                        "canonical_asset_sha256": asset_sha,
                                        # A registry-wide manifest may evolve
                                        # while this exact PLY stays unchanged.
                                        "exact_asset_manifest_sha256": "c" * 64,
                                    }
                                ]
                            }
                            if ordinal == 0
                            else {"selected_obj_ids": ["obj-one"]}
                        ),
                        "content_sha256": hashes,
                    }
                )
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "scene_name": "scene-0003",
                        "frame_index": 13,
                        "camera_id": "0",
                        "camera_name": "CAM_FRONT",
                        "camera_exposure_timestamp_us": 1000 + ordinal * 100,
                        "selected_obj_ids": ["obj-one"],
                        "content_sha256": hashes,
                        "pair_contract": {
                            "storm_checkpoint_sha256": checkpoint_hashes["storm"],
                            "cvac_checkpoint_sha256": checkpoint_hashes["cvac"],
                            "dcn_checkpoint_sha256": checkpoint_hashes["dcn"],
                        },
                        "quality": {"quality_gate_pass": True},
                    }
                )
            records_path = root / "records.jsonl"
            metadata_path = root / "metadata.jsonl"
            atomic_jsonl(records_path, records)
            atomic_jsonl(metadata_path, metadata)

            jobs = []
            cases = []
            sample_data = []
            for ordinal, context in enumerate(contexts):
                visibility = {
                    f"000013:c{camera}": camera in {"0", "2"}
                    for camera, _ in CAMERA_RING
                }
                jobs.append(
                    {
                        "multi_case_id": context,
                        "multi_case_index": ordinal,
                        "scene_name": "scene-0003",
                        "obj_id": "obj-one",
                        "global_uid": "asset-one",
                        "instance_token": "token-one",
                        "category": "car",
                        "exact_asset_binding": {
                            "global_uid": "asset-one",
                            "obj_id": "obj-one",
                            "instance_token": "token-one",
                            "category": "car",
                            "asset_path": str(root / "asset.ply"),
                            "asset_sha256": asset_sha,
                            "manifest_sha256": manifest_sha,
                            "canonical": {"front_axis": "+X"},
                        },
                        "target_usable_by_frame_camera": visibility,
                    }
                )
                cameras = {}
                for camera, camera_name in CAMERA_RING:
                    timestamp = 1000 + ordinal * 100 + int(camera)
                    cameras[camera] = {
                        "visible": camera in {"0", "2"},
                        "camera_id": camera,
                        "camera_name": camera_name,
                        "camera_exposure_timestamp_us": timestamp,
                        "sample_data_timestamp_us": timestamp,
                        "image_path": f"/raw/{camera_name}/{timestamp}.jpg",
                        "camera_depth_center_m": 10.0,
                        "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                    }
                    sample_token = (
                        "sample-bad-other"
                        if ordinal == 1 and camera == "5"
                        else f"sample-{ordinal}"
                    )
                    sample_data.append(
                        {
                            "token": f"sample-data-{ordinal}-{camera}",
                            "sample_token": sample_token,
                            "timestamp": timestamp,
                            "filename": f"sweeps/{camera_name}/x__{camera_name}__{timestamp}.jpg",
                        }
                    )
                cases.append(
                    {
                        "case_id": context,
                        "scene_name": "scene-0003",
                        "scene_index": 3,
                        "scene_token": "scene-token-three",
                        "storm_context_start": 10,
                        "storm_context_frames": [10, 15, 20, 25],
                        "storm_target_frames": [13, 17, 21, 24],
                        "storm_target_offsets": [3, 7, 11, 14],
                        "camera_ids": ["1", "0", "2", "4", "5", "3"],
                        "assets": [
                            {
                                "global_uid": "asset-one",
                                "obj_id": "obj-one",
                                "instance_token": "token-one",
                                "official_trajectory": {
                                    "frames": [
                                        {
                                            "frame_index": 13,
                                            "sample_annotation_token": "annotation-one",
                                            "translation_global_m": [1.0, 2.0, 3.0],
                                            "size_wlh_m": [1.8, 4.2, 1.6],
                                            "rotation_global_quaternion_wxyz": [
                                                1.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                            ],
                                            "heading_global_xyz": [1.0, 0.0, 0.0],
                                            "cameras": cameras,
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                )
            visibility_path = root / "visibility.json"
            trajectory_path = root / "trajectory.json"
            sample_data_path = root / "sample_data.json"
            official_scenes = root / "official_scenes.json"
            atomic_json(visibility_path, {"jobs": jobs})
            atomic_json(trajectory_path, {"schema_version": 1, "cases": cases})
            atomic_jsonl(root / "unused.jsonl", [])
            sample_data_path.write_text(json.dumps(sample_data), encoding="utf-8")
            atomic_json(official_scenes, {"train": [], "val": ["scene-0003"]})

            summary = build_strict_sixcam_index(
                split="val",
                source_root=source,
                records_path=records_path,
                metadata_paths=[metadata_path],
                visibility_manifest=visibility_path,
                trajectory_plan=trajectory_path,
                sample_data_table=sample_data_path,
                official_scenes=official_scenes,
                render_contract=render_contract,
                output_root=root / "index",
            )
            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["source_lineage_counts"], {"complete": 2})
            self.assertEqual(summary["candidate_group_count"], 1)
            self.assertEqual(summary["ready_group_count"], 0)
            self.assertEqual(summary["excluded_group_count"], 1)
            self.assertEqual(summary["baseline_context_job_count"], 1)
            self.assertEqual(summary["missing_baseline_view_count"], 5)
            self.assertEqual(summary["missing_visible_view_count"], 1)

            baseline_job = json.loads(
                (root / "index/baseline_jobs.jsonl").read_text()
            )
            self.assertEqual(
                baseline_job["trajectory_plan"]["render_context"]["case_id"],
                contexts[0],
            )
            self.assertEqual(
                baseline_job["trajectory_plan"]["render_context"][
                    "storm_target_frames"
                ],
                [13, 17, 21, 24],
            )
            visible_job = json.loads(
                (root / "index/visible_backfill_jobs.jsonl").read_text()
            )
            self.assertEqual(visible_job["job_kind"], "sixcam_visible_backfill")
            self.assertEqual(visible_job["selected_obj_ids"], ["obj-one"])
            self.assertEqual(
                visible_job["assets"][0]["official_dimensions_m"], [4.2, 1.8, 1.6]
            )
            self.assertEqual(visible_job["assets"][0]["asset_sha256"], asset_sha)
            self.assertEqual(visible_job["window_id"], contexts[0])

            group = json.loads((root / "index/groups.jsonl").read_text())
            self.assertEqual(group["context_id"], contexts[0])
            self.assertEqual(group["sample_token"], "sample-0")
            self.assertTrue(group["group_id"].startswith("val__scene-0003__f000013"))
            self.assertEqual(
                [view["camera_id"] for view in group["views"]],
                [camera for camera, _ in CAMERA_RING],
            )
            visible = {
                view["camera_id"] for view in group["views"] if view["visible"]
            }
            self.assertEqual(visible, {"0", "2"})
            for view in group["views"]:
                self.assertEqual(view["no_op"], not view["visible"])
            excluded = json.loads((root / "index/excluded_groups.jsonl").read_text())
            self.assertIn("cross official sample_token", excluded["reason"])


if __name__ == "__main__":
    unittest.main()
