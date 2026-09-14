from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from driveharm.contracts import (
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    sha256_file,
)
from driveharm.sixcam import (
    CAMERA_RING,
    audit_sixcam_release,
    audit_strict_sixcam_release,
    build_strict_sixcam_review_sheets,
    build_sixcam_release,
    publish_strict_sixcam_release,
)


class SixCameraReleaseTests(unittest.TestCase):
    def test_strict_root_release_and_inode_aware_reaudit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_root = root / "sources"
            source_root.mkdir()
            checkpoint_hashes = {
                "storm": "a" * 64,
                "cvac": "b" * 64,
                "dcn": "c" * 64,
            }
            production_record = root / "production-record.json"
            production_record.write_text("{}\n", encoding="utf-8")
            production_manifest = root / "production-metadata.jsonl"
            production_manifest.write_text("{}\n", encoding="utf-8")
            asset = {
                "global_uid": "asset-one",
                "obj_id": "obj-one",
                "instance_token": "token-one",
                "category": "car",
                "canonical_asset_path": "/asset.ply",
                "canonical_asset_sha256": "d" * 64,
                "exact_asset_manifest_sha256": "e" * 64,
                "forward_axis": "+X",
            }
            sources = []
            exposures = {}
            for ordinal, (camera, camera_name) in enumerate(CAMERA_RING):
                sample_id = f"source-c{camera}"
                paths = {}
                hashes = {}
                for role_index, role in enumerate(("gt", "input", "target")):
                    value = 20 + ordinal * 20 + role_index
                    if role == "input" and camera == "0":
                        value += 20
                    path = source_root / f"{sample_id}__{role}.png"
                    Image.fromarray(
                        np.full((288, 512, 3), value, dtype=np.uint8), mode="RGB"
                    ).save(path)
                    paths[role] = str(path)
                    hashes[role] = sha256_file(path)
                exposure = {
                    "camera_id": camera,
                    "camera_name": camera_name,
                    "camera_exposure_timestamp_us": 1000 + ordinal,
                    "sample_data_timestamp_us": 1000 + ordinal,
                    "image_path": f"/raw/{camera_name}.jpg",
                    "sample_data_token": f"sample-data-{camera}",
                    "sample_token": "sample-token-one",
                    "official_filename": f"samples/{camera_name}/x.jpg",
                }
                exposures[camera] = exposure
                row = {
                    "schema_version": 1,
                    "source_kind": "accepted_nusc_pair",
                    "split": "train",
                    "sample_id": sample_id,
                    "context_id": "scene-0001__w000010__111111111111",
                    "scene_name": "scene-0001",
                    "frame_index": 13,
                    "camera_id": camera,
                    "camera_name": camera_name,
                    "combination_id": "combination-one",
                    "selected_asset_ids": ["asset-one"],
                    "assets": [asset],
                    "checkpoint_sha256": checkpoint_hashes,
                    "exposure": exposure,
                    "paths": paths,
                    "content_sha256": hashes,
                    "lineage_status": "complete",
                    "authority": {"row_sha256": "f" * 64},
                    "production_metadata": {
                        "line_number": 1,
                        "manifest": str(production_manifest),
                        "manifest_sha256": sha256_file(production_manifest),
                        "row_sha256": "1" * 64,
                    },
                    "production_record": str(production_record),
                    "quality_gate_pass": True,
                }
                row["record_sha256"] = canonical_sha256(row)
                sources.append(row)
            identity = {
                "schema_version": 1,
                "split": "train",
                "context_id": "scene-0001__w000010__111111111111",
                "scene_name": "scene-0001",
                "sample_token": "sample-token-one",
                "frame_index": 13,
                "asset_union": [
                    {
                        "global_uid": "asset-one",
                        "obj_id": "obj-one",
                        "instance_token": "token-one",
                        "canonical_asset_sha256": "d" * 64,
                    }
                ],
                "checkpoint_sha256": checkpoint_hashes,
            }
            identity_sha = canonical_sha256(identity)
            group = {
                "schema_version": 2,
                "status": "ready",
                "group_id": f"train__scene-0001__f000013__g-{identity_sha[:16]}",
                "group_identity": identity,
                "group_identity_sha256": identity_sha,
                "split": "train",
                "context_id": identity["context_id"],
                "scene_name": "scene-0001",
                "sample_token": "sample-token-one",
                "frame_index": 13,
                "asset_union": [asset],
                "asset_state_sha256": {"asset-one": "2" * 64},
                "views": [],
            }
            for camera, camera_name in CAMERA_RING:
                visible = camera == "0"
                group["views"].append(
                    {
                        "camera_id": camera,
                        "camera_name": camera_name,
                        "status": "ready",
                        "visible": visible,
                        "no_op": not visible,
                        "no_op_reason": None if visible else "frozen visibility says absent",
                        "camera_visible_asset_ids": ["asset-one"] if visible else [],
                        "camera_invisible_asset_ids": [] if visible else ["asset-one"],
                        "asset_state_sha256": (
                            {"asset-one": "2" * 64} if visible else {}
                        ),
                        "exposure": exposures[camera],
                        "baseline_source_sample_id": f"source-c{camera}",
                        "pair_source_sample_id": "source-c0" if visible else None,
                    }
                )
            group["record_sha256"] = canonical_sha256(group)

            index = root / "index"
            index.mkdir()
            files = {
                "source_index": index / "source_index.jsonl",
                "groups": index / "groups.jsonl",
                "baseline_jobs": index / "baseline_jobs.jsonl",
                "visible_backfill_jobs": index / "visible_backfill_jobs.jsonl",
            }
            atomic_jsonl(files["source_index"], sources)
            atomic_jsonl(files["groups"], [group])
            atomic_jsonl(files["baseline_jobs"], [])
            atomic_jsonl(files["visible_backfill_jobs"], [])
            atomic_json(
                index / "summary.json",
                {
                    "status": "complete",
                    "split": "train",
                    "content_verification": {"performed": True},
                    "outputs": {
                        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                        for name, path in files.items()
                    },
                },
            )
            destination = root / "strict-release"
            summary = publish_strict_sixcam_release(
                index_roots=[index],
                destination=destination,
                receipt_root=root / "receipt",
                materialize="hardlink",
            )
            self.assertEqual(summary["group_count"], 1)
            self.assertEqual(summary["image_count"], 18)
            self.assertEqual(len(list((destination / "train").glob("*.png"))), 18)
            self.assertEqual(len(list((destination / "val").glob("*.png"))), 0)
            manifest = json.loads(
                (destination / "metadata/train_groups.jsonl").read_text()
            )
            self.assertEqual(manifest["review"]["status"], "accepted")
            for view in manifest["views"]:
                for role in ("gt", "input", "target"):
                    expected = (
                        f"train/{manifest['group_id']}__{view['camera_name']}__"
                        f"{role}.png"
                    )
                    self.assertEqual(
                        view["role_sources"][role]["published_relative_path"],
                        expected,
                    )
                if view["no_op"]:
                    input_path = destination / "train" / (
                        f"{manifest['group_id']}__{view['camera_name']}__input.png"
                    )
                    target_path = destination / "train" / (
                        f"{manifest['group_id']}__{view['camera_name']}__target.png"
                    )
                    self.assertTrue(input_path.samefile(target_path))
            self.assertTrue(
                (destination / "metadata/source_index/train/source_index.jsonl").is_file()
            )
            self.assertTrue(
                (destination / "metadata/receipts/provenance_manifest.json").is_file()
            )
            self.assertTrue(
                (destination / "metadata/reports/release_summary.json").is_file()
            )
            readme = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn("Visible edited camera views", readme)
            self.assertIn("sixcam-strict-audit", readme)
            audited = audit_strict_sixcam_release(
                destination, root / "independent-audit"
            )
            self.assertEqual(audited["status"], "pass")
            self.assertEqual(audited["logical_image_count"], 18)
            sheets = build_strict_sixcam_review_sheets(
                destination, root / "review-sheets"
            )
            self.assertEqual(sheets["group_count"], 1)
            self.assertEqual(sheets["image_count"], 18)
            sheet = next((root / "review-sheets/train").glob("*.png"))
            with Image.open(sheet) as image:
                self.assertEqual(image.size, (512 * 3, (288 + 24) * 6))

    def test_flat_eighteen_image_group_and_invisible_noop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            for role in ("gt", "input", "target"):
                (source / role).mkdir(parents=True)
            rows = []

            def save(sample: str, camera: str, selected: bool) -> dict:
                hashes = {}
                for ordinal, role in enumerate(("gt", "input", "target")):
                    value = 20 + int(camera) * 20
                    if role == "input" and selected:
                        value += 7
                    image = np.full((288, 512, 3), value, dtype=np.uint8)
                    path = source / role / f"{sample}.png"
                    Image.fromarray(image, mode="RGB").save(path)
                    hashes[role] = sha256_file(path)
                return {
                    "sample_id": sample,
                    "scene_name": "scene-0001",
                    "frame_index": 12,
                    "camera_id": camera,
                    "assets": ([{"global_uid": "asset-one"}] if selected else []),
                    "content_sha256": hashes,
                }

            for camera, _ in CAMERA_RING:
                rows.append(save(f"baseline-c{camera}", camera, False))
            rows.append(save("selected-c0", "0", True))
            rows.append(save("selected-c2", "2", True))
            records = root / "records.jsonl"
            atomic_jsonl(records, rows)
            visibility = root / "visibility.json"
            atomic_json(
                visibility,
                {
                    "jobs": [
                        {
                            "scene_name": "scene-0001",
                            "obj_id": "obj-one",
                            "global_uid": "asset-one",
                            "instance_token": "token-one",
                            "category": "car",
                            "exact_asset_binding": {
                                "global_uid": "asset-one",
                                "obj_id": "obj-one",
                                "instance_token": "token-one",
                                "category": "car",
                            },
                            "target_usable_by_frame_camera": {
                                "000012:c0": True,
                                "000012:c1": False,
                                "000012:c2": True,
                                "000012:c3": False,
                                "000012:c4": False,
                                "000012:c5": False,
                            },
                        }
                    ]
                },
            )
            destination = root / "flat"
            receipt = root / "receipt"
            summary = build_sixcam_release(
                source, records, visibility, destination, receipt, materialize="copy"
            )
            self.assertEqual(summary["group_count"], 1)
            self.assertEqual(summary["image_count"], 18)
            images = sorted(destination.glob("*.png"))
            self.assertEqual(len(images), 18)
            group = json.loads((receipt / "groups.jsonl").read_text())
            self.assertEqual(len(group["views"]), 6)
            for view in group["views"]:
                input_path = destination / view["files"]["input"]
                target_path = destination / view["files"]["target"]
                if view["camera_id"] in {"0", "2"}:
                    self.assertTrue(view["asset_visible"])
                    self.assertNotEqual(sha256_file(input_path), sha256_file(target_path))
                else:
                    self.assertFalse(view["asset_visible"])
                    self.assertEqual(sha256_file(input_path), sha256_file(target_path))
            audited = audit_sixcam_release(
                destination,
                receipt / "groups.jsonl",
                records,
                visibility,
                root / "audit",
            )
            self.assertEqual(audited["status"], "pass")
            self.assertTrue(audited["all_images_checked"])

            visibility_payload = json.loads(visibility.read_text())
            usable = visibility_payload["jobs"][0][
                "target_usable_by_frame_camera"
            ]
            usable["000012:c2"] = False
            atomic_json(visibility, visibility_payload)
            bad_binding = audit_sixcam_release(
                destination,
                receipt / "groups.jsonl",
                records,
                visibility,
                root / "audit-bad-binding",
            )
            self.assertEqual(bad_binding["status"], "fail")
            usable["000012:c2"] = True
            atomic_json(visibility, visibility_payload)

            extra = destination / "unexpected.png"
            Image.new("RGB", (512, 288)).save(extra)
            failed = audit_sixcam_release(
                destination,
                receipt / "groups.jsonl",
                records,
                visibility,
                root / "audit-bad",
            )
            self.assertEqual(failed["status"], "fail")


if __name__ == "__main__":
    unittest.main()
