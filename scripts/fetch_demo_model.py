#!/usr/bin/env python3
"""Download the pinned public demo model artifact from GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    project2_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fetch the demo model artifact from GitHub Releases")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project2_root / "mondrian_artifacts_demo" / "meta" / "model_asset.json",
        help="Path to the model asset manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the model even if it already exists locally.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=600,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--skip-checksum-asset",
        action="store_true",
        help="Skip downloading and validating the .sha256 release asset.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, output_path: Path, timeout_sec: int) -> None:
    with requests.get(url, stream=True, timeout=timeout_sec) as resp:
        resp.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _parse_checksum_file(path: Path, expected_name: str) -> str:
    text = path.read_text(encoding="utf-8").strip()
    m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", text)
    if not m:
        raise ValueError(f"Unexpected checksum file format in {path}")
    checksum = m.group(1).lower()
    asset_name = Path(m.group(2).strip()).name
    if asset_name != expected_name:
        raise ValueError(
            f"Checksum asset references '{asset_name}', expected '{expected_name}'"
        )
    return checksum


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)
    project_root = manifest_path.parents[2]

    download_url = str(manifest.get("download_url") or "").strip()
    checksum_url = str(manifest.get("checksum_url") or "").strip()
    target_rel = str(manifest.get("target_path") or "").strip()
    asset_name = str(manifest.get("asset_name") or "").strip()
    checksum_asset_name = str(manifest.get("checksum_asset_name") or "").strip()
    expected_asset_sha = str(manifest.get("asset_sha256") or "").strip().lower()
    expected_checksum_sha = str(manifest.get("checksum_asset_sha256") or "").strip().lower()

    if not download_url or not target_rel or not asset_name or not expected_asset_sha:
        raise ValueError(f"Manifest is missing required fields: {manifest_path}")

    target_path = (project_root / target_rel).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not args.force:
        actual_sha = _hash_file(target_path)
        if actual_sha == expected_asset_sha:
            print(f"[OK] Model already present and verified: {target_path}")
            return
        print(
            "[WARN] Existing model does not match manifest checksum. "
            "Re-run with --force to replace it.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tmp_model = target_path.with_suffix(target_path.suffix + ".download")
    tmp_checksum = target_path.parent / f"{checksum_asset_name}.download"
    if tmp_model.exists():
        tmp_model.unlink()
    if tmp_checksum.exists():
        tmp_checksum.unlink()

    print(f"[INFO] Downloading {asset_name} from pinned release")
    _download(download_url, tmp_model, args.timeout_sec)

    actual_asset_sha = _hash_file(tmp_model)
    if actual_asset_sha != expected_asset_sha:
        tmp_model.unlink(missing_ok=True)
        raise ValueError(
            "Model checksum mismatch after download: "
            f"expected {expected_asset_sha}, got {actual_asset_sha}"
        )
    print(f"[OK] Verified model SHA256: {actual_asset_sha}")

    if checksum_url and checksum_asset_name and not args.skip_checksum_asset:
        print(f"[INFO] Downloading checksum asset {checksum_asset_name}")
        _download(checksum_url, tmp_checksum, args.timeout_sec)

        actual_checksum_sha = _hash_file(tmp_checksum)
        if expected_checksum_sha and actual_checksum_sha != expected_checksum_sha:
            tmp_model.unlink(missing_ok=True)
            tmp_checksum.unlink(missing_ok=True)
            raise ValueError(
                "Checksum asset hash mismatch: "
                f"expected {expected_checksum_sha}, got {actual_checksum_sha}"
            )

        checksum_from_file = _parse_checksum_file(tmp_checksum, asset_name)
        if checksum_from_file != expected_asset_sha:
            tmp_model.unlink(missing_ok=True)
            tmp_checksum.unlink(missing_ok=True)
            raise ValueError(
                "Checksum file content does not match manifest model hash: "
                f"{checksum_from_file} != {expected_asset_sha}"
            )
        print(f"[OK] Verified checksum asset for {asset_name}")

    tmp_model.replace(target_path)
    if tmp_checksum.exists():
        tmp_checksum.unlink()

    print(f"[OK] Model downloaded to {target_path}")


if __name__ == "__main__":
    main()
