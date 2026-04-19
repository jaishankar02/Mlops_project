"""HR-VITON inference wrapper used as a low-VRAM fallback."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import torch

from config.settings import settings
from backend.hr_viton.get_parse_agnostic import get_im_parse_agnostic

logger = logging.getLogger(__name__)


class HRVITONWrapper:
    """Thin wrapper around a local HR-VITON checkout and its inference script.

    The wrapper expects a local clone of https://github.com/sangyun884/HR-VITON and
    pretrained checkpoints available on disk.
    """

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.enabled = settings.HR_VITON_ENABLED
        self.repo_path = Path(settings.HR_VITON_LOCAL_REPO_PATH)
        if not self.repo_path.is_absolute():
            self.repo_path = (project_root / self.repo_path).resolve()
        self.test_name = settings.HR_VITON_TEST_NAME
        if settings.HR_VITON_TOCG_CHECKPOINT:
            self.tocg_checkpoint = Path(settings.HR_VITON_TOCG_CHECKPOINT)
            if not self.tocg_checkpoint.is_absolute():
                self.tocg_checkpoint = (project_root / self.tocg_checkpoint).resolve()
        else:
            self.tocg_checkpoint = self.repo_path / "eval_models" / "weights" / "v0.1" / "mtviton.pth"

        if settings.HR_VITON_GEN_CHECKPOINT:
            self.gen_checkpoint = Path(settings.HR_VITON_GEN_CHECKPOINT)
            if not self.gen_checkpoint.is_absolute():
                self.gen_checkpoint = (project_root / self.gen_checkpoint).resolve()
        else:
            self.gen_checkpoint = self.repo_path / "eval_models" / "weights" / "v0.1" / "gen.pth"
        self.force_cpu = settings.HR_VITON_FORCE_CPU
        self.strict_official = settings.HR_VITON_STRICT_OFFICIAL
        self._cached_availability: Optional[bool] = None
        self.last_error: Optional[str] = None

    def get_last_error(self) -> Optional[str]:
        return self.last_error

    def is_available(self) -> bool:
        if self._cached_availability is not None:
            return self._cached_availability

        available = (
            self.enabled
            and torch.cuda.is_available()
            and self.repo_path.exists()
            and self.tocg_checkpoint.exists()
            and self.gen_checkpoint.exists()
        )
        self._cached_availability = available
        if not available:
            logger.info(
                "HR-VITON unavailable: repo=%s tocg=%s gen=%s",
                self.repo_path,
                self.tocg_checkpoint,
                self.gen_checkpoint,
            )
        return available

    @staticmethod
    def _image_to_pil(image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(np.asarray(image)).convert("RGB")

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _estimate_garment_mask(garment_image: Image.Image) -> Image.Image:
        rgb = garment_image.convert("RGB")
        array = np.array(rgb)
        # Treat near-white pixels as background. This is conservative for product images.
        background = np.all(array > 245, axis=2)
        mask = np.where(background, 0, 255).astype(np.uint8)
        if mask.mean() < 5:
            mask[:, :] = 255
        return Image.fromarray(mask, mode="L")

    @staticmethod
    def _make_pose_json(keypoints: dict[str, object]) -> dict:
        raw_points = keypoints.get("pose_keypoints_2d", [])
        flat_points: list[float] = []

        # CPDatasetTest expects OpenPose-style flattened [x, y, confidence] entries.
        if isinstance(raw_points, list) and raw_points and isinstance(raw_points[0], (list, tuple)):
            for point in raw_points:
                if len(point) >= 2:
                    flat_points.extend([float(point[0]), float(point[1]), 0.0])
        elif isinstance(raw_points, list):
            # Already flattened; pad every 2 coordinates with a confidence score.
            if len(raw_points) % 3 == 0:
                flat_points = [float(v) for v in raw_points]
            else:
                for i in range(0, len(raw_points), 2):
                    if i + 1 < len(raw_points):
                        flat_points.extend([float(raw_points[i]), float(raw_points[i + 1]), 0.0])

        if not flat_points:
            flat_points = [0.0] * (18 * 3)

        return {"people": [{"pose_keypoints_2d": flat_points}]}

    @staticmethod
    def _pose_to_rendered_image(keypoints: list) -> Image.Image:
        canvas = Image.new("RGB", (768, 1024), "white")
        draw = ImageDraw.Draw(canvas)
        for x, y in keypoints:
            if x and y:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0))
        return canvas

    @staticmethod
    def _build_parse_agnostic(parse_image: Image.Image, pose_data: list[list[float]]) -> Image.Image:
        # Use the repository's official-style parse-agnostic generation.
        return get_im_parse_agnostic(parse_image, np.asarray(pose_data))

    def generate_tryon(self, person_image: Image.Image, garment_image: Image.Image) -> Optional[Image.Image]:
        """Generate a try-on result by preparing an HR-VITON-compatible request and invoking its test generator.

        This path requires a local HR-VITON checkout plus checkpoints. If the repo or checkpoints are
        unavailable, returns None so the caller can fall back again.
        """
        self.last_error = None
        if not self.is_available():
            self.last_error = "HR-VITON is unavailable (repo/checkpoint/cuda prerequisites not met)."
            return None

        repo_str = str(self.repo_path)
        idm_repo_path = Path(__file__).resolve().parents[2] / "backend" / "idm_vton"
        idm_repo_str = str(idm_repo_path)
        idm_demo_str = str(idm_repo_path / "gradio_demo")
        idm_preprocess_str = str(idm_repo_path / "preprocess")
        idm_humanparsing_str = str(idm_repo_path / "preprocess" / "humanparsing")
        demo_str = str(self.repo_path / "gradio_demo")
        preprocess_str = str(self.repo_path / "preprocess")
        humanparsing_str = str(self.repo_path / "preprocess" / "humanparsing")
        for path_str in [
            idm_repo_str,
            idm_demo_str,
            idm_preprocess_str,
            idm_humanparsing_str,
            repo_str,
            demo_str,
            preprocess_str,
            humanparsing_str,
        ]:
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        try:
            from backend.idm_vton.preprocess.openpose.run_openpose import OpenPose
            from backend.idm_vton.preprocess.humanparsing.run_parsing import Parsing
            from backend.idm_vton.gradio_demo import apply_net
            from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
        except Exception as exc:
            self.last_error = f"Unable to import preprocessing stack: {exc}"
            logger.warning("Unable to import preprocessing stack for HR-VITON: %s", exc)
            return None

        person_rgb = self._image_to_pil(person_image).resize((768, 1024))
        garment_rgb = self._image_to_pil(garment_image).resize((768, 1024))

        try:
            openpose_model = OpenPose(0)
            parsing_model = Parsing(0)
        except Exception as exc:
            self.last_error = f"Failed to initialize preprocessing models: {exc}"
            logger.warning("Failed to initialize HR-VITON preprocessing models: %s", exc)
            return None

        with tempfile.TemporaryDirectory(prefix="stylesync_hr_viton_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            dataroot = tmp_root / "data"
            test_dir = dataroot / "test"
            for subdir in ["image", "cloth", "cloth-mask", "image-parse-v3", "image-parse-agnostic-v3.2", "image-densepose", "openpose_json", "openpose_img"]:
                self._ensure_dir(test_dir / subdir)
            self._ensure_dir(dataroot)

            image_name = "person.jpg"
            cloth_name = "garment.jpg"
            # HR-VITON test loader parses lines as "im_name cloth_name".
            pair_name = f"{image_name} {cloth_name}\n"

            person_rgb.save(test_dir / "image" / image_name)
            garment_rgb.save(test_dir / "cloth" / cloth_name)
            self._estimate_garment_mask(garment_rgb).save(test_dir / "cloth-mask" / cloth_name)
            # CPDatasetTest loads both paired and unpaired cloth entries during __getitem__,
            # so provide the person-named cloth alias to avoid missing-file errors.
            garment_rgb.save(test_dir / "cloth" / image_name)
            self._estimate_garment_mask(garment_rgb).save(test_dir / "cloth-mask" / image_name)

            try:
                keypoints = openpose_model(person_rgb.resize((384, 512)))
                pose_data = keypoints.get("pose_keypoints_2d", [])
                if len(pose_data) < 18:
                    pose_data = [[0.0, 0.0] for _ in range(18)]
                    keypoints = {"pose_keypoints_2d": pose_data}
            except Exception as exc:
                # Some images do not produce OpenPose detections; continue with empty keypoints.
                logger.warning("OpenPose preprocessing fallback for HR-VITON: %s", exc)
                pose_data = [[0.0, 0.0] for _ in range(18)]
                keypoints = {"pose_keypoints_2d": pose_data}

            pose_image = self._pose_to_rendered_image(pose_data)
            pose_json_path = test_dir / "openpose_json" / image_name.replace(".jpg", "_keypoints.json")
            with open(pose_json_path, "w", encoding="utf-8") as handle:
                json.dump(self._make_pose_json(keypoints), handle)
            pose_image.save(test_dir / "openpose_img" / image_name.replace(".jpg", "_rendered.png"))

            try:
                parse_image, _ = parsing_model(person_rgb.resize((384, 512)))
                if not isinstance(parse_image, Image.Image):
                    parse_image = Image.fromarray(np.asarray(parse_image))
                # Preserve class indices from parser output; avoid RGB->L conversion that corrupts labels.
                parse_array = np.asarray(parse_image)
                if parse_array.ndim == 3:
                    parse_array = parse_array[:, :, 0]
                parse_array = np.where((parse_array >= 0) & (parse_array < 20), parse_array, 0).astype(np.uint8)
                parse_image = Image.fromarray(parse_array, mode="L").resize((768, 1024), Image.Resampling.NEAREST)
                parse_image.save(test_dir / "image-parse-v3" / image_name.replace(".jpg", ".png"))
                agnostic_image = self._build_parse_agnostic(parse_image, pose_data)
                agnostic_image.save(test_dir / "image-parse-agnostic-v3.2" / image_name.replace(".jpg", ".png"))
            except Exception as exc:
                self.last_error = f"Parsing/agnostic preprocessing failed: {exc}"
                logger.warning("Parsing/agnostic preprocessing failed for HR-VITON: %s", exc)
                return None

            try:
                human_img_arg = _apply_exif_orientation(person_rgb.resize((384, 512)))
                human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
                args = apply_net.create_argument_parser().parse_args(
                    (
                        "show",
                        str(idm_repo_path / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"),
                        str(idm_repo_path / "ckpt" / "densepose" / "model_final_162be9.pkl"),
                        "dp_segm",
                        "-v",
                        "--opts",
                        "MODEL.DEVICE",
                        "cpu" if self.force_cpu else "cuda",
                    )
                )
                pose_arr = args.func(args, human_img_arg)
                pose_arr = pose_arr[:, :, ::-1]
                pose_img = Image.fromarray(np.uint8(pose_arr)).resize((768, 1024))
                pose_img.save(test_dir / "image-densepose" / image_name)
            except Exception as exc:
                self.last_error = f"DensePose preprocessing failed: {exc}"
                logger.warning("DensePose preprocessing failed for HR-VITON: %s", exc)
                return None

            with open(dataroot / "test_pairs.txt", "w", encoding="utf-8") as handle:
                handle.write(pair_name)

            output_dir = tmp_root / "output"
            python_bin = os.getenv("HR_VITON_PYTHON_BIN", sys.executable)
            compat_path = str(Path(__file__).resolve().parent / "hr_viton_compat")
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{compat_path}:{existing_pythonpath}" if existing_pythonpath else compat_path
            cmd = [
                python_bin,
                "test_generator.py",
                "--occlusion",
                "--test_name",
                self.test_name,
                "--gpu_ids",
                "" if self.force_cpu else "0",
                "--cuda",
                "False" if self.force_cpu else "True",
                "--tocg_checkpoint",
                str(self.tocg_checkpoint),
                "--gen_checkpoint",
                str(self.gen_checkpoint),
                "--datasetting",
                "unpaired",
                "--dataroot",
                str(dataroot),
                "--data_list",
                "test_pairs.txt",
                "--output_dir",
                str(output_dir),
                "--batch-size",
                "1",
                "--workers",
                "0",
            ]

            try:
                subprocess.run(
                    cmd,
                    cwd=str(self.repo_path),
                    env=env,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                self.last_error = f"HR-VITON subprocess failed: {exc.stderr[-400:] if exc.stderr else exc}"
                logger.warning("HR-VITON subprocess failed: %s", exc.stderr[-2000:] if exc.stderr else exc)
                return None

            candidates = list(output_dir.rglob("*.png")) + list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.jpeg"))
            if not candidates:
                self.last_error = "HR-VITON finished without producing an image."
                logger.warning("HR-VITON did not produce an output image")
                return None

            return Image.open(candidates[0]).convert("RGB")


_hr_viton_instance: Optional[HRVITONWrapper] = None


def get_hr_viton_wrapper() -> HRVITONWrapper:
    global _hr_viton_instance
    if _hr_viton_instance is None:
        _hr_viton_instance = HRVITONWrapper()
    return _hr_viton_instance
