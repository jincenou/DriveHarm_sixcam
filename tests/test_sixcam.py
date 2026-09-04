from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from driveharm.contracts import atomic_json, atomic_jsonl, sha256_file
from driveharm.sixcam import CAMERA_RING, audit_sixcam_release, build_sixcam_release


class SixCameraReleaseTests(unittest.TestCase):
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
