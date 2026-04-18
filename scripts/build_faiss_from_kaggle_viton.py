#!/usr/bin/env python3
"""Download Kaggle VITON-Zalando dataset and build FAISS similarity index.

Example:
  .venv/bin/python3 scripts/build_faiss_from_kaggle_viton.py \
    --dataset marquis03/high-resolution-viton-zalando-dataset \
    --index-path data/faiss_index.bin \
    --limit 5000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image

from config.settings import settings
from ml_models.recommender.recommendation_engine import RecommendationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kaggle VITON dataset and build FAISS index from cloth images"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="marquis03/high-resolution-viton-zalando-dataset",
        help="Kaggle dataset id owner/name",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="data/kaggle_downloads",
        help="Where Kaggle zip files are downloaded",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data/datasets/viton_zalando_hr",
        help="Where the dataset is extracted",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=settings.FAISS_INDEX_PATH,
        help="FAISS index output path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum images to index (0 means all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download and use existing extracted files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="clip",
        choices=["clip", "resnet"],
        help="Feature extractor backend",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract zip even if extraction marker exists",
    )
    return parser.parse_args()


def _check_kaggle_auth() -> Tuple[bool, str]:
    username = (settings.KAGGLE_USERNAME or "").strip()
    key = (settings.KAGGLE_KEY or "").strip()

    if username and key:
        return True, "Kaggle auth found in settings/.env"

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            content = json.loads(kaggle_json.read_text(encoding="utf-8"))
            if content.get("username") and content.get("key"):
                return True, f"Kaggle auth found in {kaggle_json}"
        except Exception:
            pass

    return False, (
        "Missing Kaggle auth. Set KAGGLE_USERNAME and KAGGLE_KEY in .env "
        "or create ~/.kaggle/kaggle.json"
    )


def _run_kaggle_download(dataset: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    kaggle_cli = str(Path(sys.executable).with_name("kaggle"))
    if not Path(kaggle_cli).exists():
        kaggle_cli = "kaggle"
    cmd = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(download_dir),
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    slug = dataset.split("/", 1)[1]
    expected_zip = download_dir / f"{slug}.zip"
    if expected_zip.exists():
        return expected_zip

    zips = sorted(download_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise RuntimeError("Kaggle reported success but no zip file was found in download dir")
    return zips[0]


def _extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extract_complete"

    if marker.exists() and not force:
        return

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    marker.write_text(f"source_zip={zip_path}\n", encoding="utf-8")


def _discover_cloth_dirs(root: Path) -> List[Path]:
    splits = {"train", "test", "val", "valid", "validation"}
    cloth_dirs = []
    for path in root.rglob("cloth"):
        if not path.is_dir():
            continue
        parent = path.parent.name.lower()
        if parent in splits:
            cloth_dirs.append(path)

    unique_dirs = sorted(set(cloth_dirs))
    if unique_dirs:
        return unique_dirs

    fallback = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "cloth"]
    return sorted(set(fallback))


def _iter_images(cloth_dirs: Iterable[Path]) -> Iterable[Tuple[Path, str]]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for cloth_dir in cloth_dirs:
        split = cloth_dir.parent.name
        for image in sorted(cloth_dir.rglob("*")):
            if image.is_file() and image.suffix.lower() in exts:
                yield image, split


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir)
    extract_dir = Path(args.extract_dir)
    index_path = Path(args.index_path)

    print(f"dataset={args.dataset}")
    print(f"download_dir={download_dir}")
    print(f"extract_dir={extract_dir}")
    print(f"index_path={index_path}")

    if not args.skip_download:
        auth_ok, msg = _check_kaggle_auth()
        if not auth_ok:
            raise SystemExit(msg)
        print(msg)

        zip_path = _run_kaggle_download(args.dataset, download_dir)
        print(f"downloaded_zip={zip_path}")

        _extract_zip(zip_path, extract_dir, force=args.force_extract)
        print("extraction_complete=true")

    if not extract_dir.exists():
        raise SystemExit(f"Extract directory not found: {extract_dir}")

    cloth_dirs = _discover_cloth_dirs(extract_dir)
    if not cloth_dirs:
        raise SystemExit(
            "Could not find cloth directories in extracted dataset. "
            f"Checked recursively under {extract_dir}."
        )

    print("cloth_dirs=")
    for d in cloth_dirs:
        print(f"  - {d}")

    all_images = list(_iter_images(cloth_dirs))
    if args.limit > 0:
        all_images = all_images[: args.limit]

    print(f"total_images_to_index={len(all_images)}")
    if not all_images:
        raise SystemExit("No images found to index")

    index_path.parent.mkdir(parents=True, exist_ok=True)

    engine = RecommendationEngine(feature_dim=512, faiss_index_path=str(index_path))
    engine.initialize_feature_extractor(model_name=args.model_name)

    added = 0
    failed = 0

    for i, (img_path, split) in enumerate(all_images, 1):
        try:
            with Image.open(img_path) as pil:
                image = pil.convert("RGB")
            ok = engine.add_garment(
                garment_id=f"kaggle_viton_{split}_{img_path.stem}_{i}",
                garment_image=image,
                metadata={
                    "source": "kaggle",
                    "dataset": args.dataset,
                    "split": split,
                    "filename": img_path.name,
                    "image_path": str(img_path),
                },
            )
            if ok:
                added += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if i % 200 == 0:
            print(f"progress={i}/{len(all_images)} added={added} failed={failed}")

    engine.save_index(str(index_path))
    print(f"index_saved={index_path}")
    print(f"metadata_saved={index_path.with_suffix('.json')}")
    print(f"completed added={added} failed={failed}")


if __name__ == "__main__":
    main()
