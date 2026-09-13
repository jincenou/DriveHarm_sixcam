"""Shared contracts and deterministic filesystem primitives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping


ROLES = ("gt", "input", "target")
CAMERAS = tuple(str(index) for index in range(6))
IMAGE_SIZE = (512, 288)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            yield value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write JSONL without materializing the manifest in memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def indexed_rows(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        value = str(row.get(key) or "")
        if not value or value in result:
            raise ValueError(f"empty or duplicate {key}: {value!r}")
        result[value] = row
    if not result:
        raise ValueError(f"empty manifest: {path}")
    return result


@dataclass(frozen=True)
class AssetBinding:
    asset_id: str
    obj_id: str
    instance_token: str
    category: str
    ply_path: Path
    ply_sha256: str
    dimensions_m: tuple[float, float, float]
    forward_axis: str

    @classmethod
    def parse(
        cls,
        row: Mapping[str, Any],
        manifest_root: Path | None = None,
        verify_content: bool = True,
    ) -> "AssetBinding":
        canonical = row.get("canonical") or {}
        dimensions = tuple(
            float(value)
            for value in (row.get("dimensions_m") or row.get("size_xyz") or ())
        )
        raw_path = Path(str(row.get("ply_path") or row.get("asset_path") or ""))
        if not raw_path.is_absolute() and manifest_root is not None:
            raw_path = manifest_root / raw_path
        binding = cls(
            asset_id=str(row.get("asset_id") or row.get("global_uid") or ""),
            obj_id=str(row.get("obj_id") or ""),
            instance_token=str(row.get("instance_token") or ""),
            category=str(row.get("category") or ""),
            ply_path=raw_path.resolve(),
            ply_sha256=str(
                row.get("ply_sha256")
                or row.get("asset_sha256")
                or canonical.get("canonical_asset_sha256")
                or ""
            ),
            dimensions_m=dimensions,
            forward_axis=str(
                row.get("forward_axis") or canonical.get("front_axis") or ""
            ),
        )
        binding.validate(verify_content)
        return binding

    def validate(self, verify_content: bool = True) -> None:
        if not all((self.asset_id, self.obj_id, self.instance_token, self.category)):
            raise ValueError("asset identity binding is incomplete")
        if len(self.dimensions_m) != 3 or any(
            value <= 0 for value in self.dimensions_m
        ):
            raise ValueError(f"invalid asset dimensions: {self.asset_id}")
        if self.forward_axis != "+X":
            raise ValueError(f"canonical asset must face +X: {self.asset_id}")
        if len(self.ply_sha256) != 64:
            raise ValueError(f"asset content hash is invalid: {self.asset_id}")
        if verify_content and sha256_file(self.ply_path) != self.ply_sha256:
            raise ValueError(f"asset content hash mismatch: {self.asset_id}")


def load_asset_post(post_path: Path) -> dict[str, AssetBinding]:
    """Load the immutable upstream asset hand-off and verify every PLY binding."""
    post = read_json(post_path.resolve(strict=True))
    if post.get("status") != "complete":
        raise ValueError("asset hand-off is not complete")
    manifest_value = post.get("manifest")
    manifest_hash = post.get("manifest_sha256")
    native_output = (post.get("outputs") or {}).get("exact_asset_manifest")
    if not manifest_value and isinstance(native_output, dict):
        manifest_value = native_output.get("path")
        manifest_hash = native_output.get("sha256")
    if not manifest_value and post.get("exact_asset_manifest"):
        manifest_value = post.get("exact_asset_manifest")
        manifest_hash = post.get("exact_asset_manifest_sha256")
    raw_manifest = Path(str(manifest_value or ""))
    if not raw_manifest.is_absolute():
        raw_manifest = post_path.resolve().parent / raw_manifest
    manifest = raw_manifest.resolve(strict=True)
    if str(manifest_hash or "") != sha256_file(manifest):
        raise ValueError("asset hand-off manifest hash mismatch")
    if manifest.suffix.lower() == ".json":
        payload = read_json(manifest)
        rows = payload.get("assets") or []
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("exact asset manifest has no asset rows")
    else:
        rows = list(iter_jsonl(manifest))
    excluded = {str(value) for value in post.get("excluded_asset_ids") or []}
    if post.get("quarantine_manifest"):
        raw_quarantine = Path(str(post["quarantine_manifest"]))
        if not raw_quarantine.is_absolute():
            raw_quarantine = post_path.resolve().parent / raw_quarantine
        quarantine = raw_quarantine.resolve(strict=True)
        if sha256_file(quarantine) != str(post.get("quarantine_manifest_sha256") or ""):
            raise ValueError("asset quarantine manifest hash mismatch")
        values = read_json(quarantine).get("assets") or []
        excluded.update(
            str(value.get("obj_id") or value.get("global_uid") or "")
            if isinstance(value, dict)
            else str(value)
            for value in values
        )
    row_ids = {
        str(row.get("obj_id") or row.get("global_uid") or row.get("asset_id") or "")
        for row in rows
    }
    if "" in excluded or not excluded.issubset(row_ids):
        raise ValueError("asset exclusions are empty or outside the exact manifest")
    rows = [
        row
        for row in rows
        if str(row.get("obj_id") or row.get("global_uid") or row.get("asset_id") or "")
        not in excluded
    ]
    bindings = [AssetBinding.parse(row, manifest.parent, False) for row in rows]

    def verify(binding: AssetBinding) -> None:
        if sha256_file(binding.ply_path) != binding.ply_sha256:
            raise ValueError(f"asset content hash mismatch: {binding.asset_id}")

    with ThreadPoolExecutor(max_workers=min(32, max(1, len(bindings)))) as executor:
        list(executor.map(verify, bindings))
    by_obj = {binding.obj_id: binding for binding in bindings}
    if len(by_obj) != len(bindings):
        raise ValueError("asset hand-off contains duplicate obj_id values")
    return by_obj
