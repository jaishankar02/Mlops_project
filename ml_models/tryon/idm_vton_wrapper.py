"""IDM-VTON integration wrapper.

This wrapper uses the pre-cloned external/IDM-VTON repository to run the official pipeline.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from config.settings import settings

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class IDMVTONWrapper:
    """IDM-VTON wrapper using external/IDM-VTON repository."""

    def __init__(self) -> None:
        self.enabled = settings.IDM_VTON_ENABLED
        self.model_id = os.getenv("IDM_VTON_MODEL_ID", settings.IDM_VTON_PRETRAINED_MODEL_NAME_OR_PATH)
        self.cache_dir = Path(os.getenv("IDM_VTON_CACHE_DIR", "/data/m25csa007/models/hf_cache"))
        self.use_official_demo = _env_flag("IDM_VTON_USE_OFFICIAL_DEMO", True)

        self.denoise_steps = int(os.getenv("IDM_VTON_STEPS", "30"))
        self.seed_value = int(os.getenv("IDM_VTON_SEED", "42"))
        self.garment_des = os.getenv("IDM_VTON_GARMENT_DESCRIPTION", "Short Sleeve Round Neck T-shirts")
        self.auto_mask = _env_flag("IDM_VTON_AUTO_MASK", True)
        self.use_crop = _env_flag("IDM_VTON_USE_CROP", False)

        self.repo_path = Path(__file__).parent.parent.parent / "external" / "IDM-VTON"
        self._demo_instance = None
        self._snapshot_path = None

    @property
    def demo_path(self) -> Path:
        return self.repo_path

    def is_available(self) -> bool:
        return bool(self.enabled and self.use_official_demo and self.repo_path.exists())

    def _load_demo(self):
        """Load the official demo from the external repo."""
        if self._demo_instance is not None:
            return self._demo_instance

        if not self.repo_path.exists():
            logger.warning("IDM-VTON repo not found at %s", self.repo_path)
            return None

        try:
            # Add repo to path for imports
            repo_str = str(self.repo_path)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)
            demo_str = str(self.repo_path / "gradio_demo")
            if demo_str not in sys.path:
                sys.path.insert(0, demo_str)
            humanparsing_str = str(self.repo_path / "preprocess" / "humanparsing")
            if humanparsing_str not in sys.path:
                sys.path.insert(0, humanparsing_str)
            preprocess_str = str(self.repo_path / "preprocess")
            if preprocess_str not in sys.path:
                sys.path.insert(0, preprocess_str)
            # IDM-VTON human parsing expects its own "utils" package; remove any preloaded project utils.
            if "utils" in sys.modules:
                del sys.modules["utils"]

            from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
            from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
            from src.unet_hacked_tryon import UNet2DConditionModel
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
                CLIPTextModel,
                CLIPTextModelWithProjection,
                AutoTokenizer,
            )
            from diffusers import DDPMScheduler, AutoencoderKL
            from utils_mask import get_mask_location
            import apply_net
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose
            from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            base_path = self.model_id
            logger.info("Loading IDM-VTON models from %s", base_path)

            unet = UNet2DConditionModel.from_pretrained(
                base_path, subfolder="unet", torch_dtype=torch_dtype
            )
            unet.requires_grad_(False)

            tokenizer_one = AutoTokenizer.from_pretrained(
                base_path, subfolder="tokenizer", use_fast=False
            )
            tokenizer_two = AutoTokenizer.from_pretrained(
                base_path, subfolder="tokenizer_2", use_fast=False
            )

            noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

            text_encoder_one = CLIPTextModel.from_pretrained(
                base_path, subfolder="text_encoder", torch_dtype=torch_dtype
            )
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                base_path, subfolder="text_encoder_2", torch_dtype=torch_dtype
            )

            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                base_path, subfolder="image_encoder", torch_dtype=torch_dtype
            )

            vae = AutoencoderKL.from_pretrained(
                base_path, subfolder="vae", torch_dtype=torch_dtype
            )

            UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
                base_path, subfolder="unet_encoder", torch_dtype=torch_dtype
            )

            pipe = TryonPipeline.from_pretrained(
                base_path,
                unet=unet,
                vae=vae,
                feature_extractor=CLIPImageProcessor(),
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                scheduler=noise_scheduler,
                image_encoder=image_encoder,
                torch_dtype=torch_dtype,
            )
            pipe.unet_encoder = UNet_Encoder

            parsing_model = Parsing(0)
            openpose_model = OpenPose(0)
            tensor_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            self._demo_instance = {
                "pipe": pipe,
                "device": device,
                "torch_dtype": torch_dtype,
                "parsing_model": parsing_model,
                "openpose_model": openpose_model,
                "tensor_transform": tensor_transform,
                "get_mask_location": get_mask_location,
                "apply_net": apply_net,
                "convert_pil_to_numpy": convert_PIL_to_numpy,
                "apply_exif_orientation": _apply_exif_orientation,
            }
            logger.info("IDM-VTON models loaded successfully")
            return self._demo_instance

        except Exception as exc:
            logger.warning("Failed to load IDM-VTON demo: %s", exc)
            return None

    def generate_tryon(self, person_image: Image.Image, garment_image: Image.Image) -> Optional[Image.Image]:
        """Generate try-on result using official IDM-VTON demo flow."""
        if not self.is_available():
            logger.warning("IDM-VTON unavailable")
            return None

        demo = self._load_demo()
        if demo is None:
            return None

        try:
            device = demo["device"]
            pipe = demo["pipe"]
            torch_dtype = demo["torch_dtype"]
            parsing_model = demo["parsing_model"]
            openpose_model = demo["openpose_model"]
            tensor_transform = demo["tensor_transform"]
            get_mask_location = demo["get_mask_location"]
            apply_net = demo["apply_net"]
            convert_pil_to_numpy = demo["convert_pil_to_numpy"]
            apply_exif_orientation = demo["apply_exif_orientation"]

            if device == "cuda":
                openpose_model.preprocessor.body_estimation.model.to(device)
            pipe.to(device)
            pipe.unet_encoder.to(device)

            garment_rgb = garment_image.convert("RGB").resize((768, 1024))
            human_rgb = person_image.convert("RGB").resize((768, 1024))

            keypoints = openpose_model(human_rgb.resize((384, 512)))
            model_parse, _ = parsing_model(human_rgb.resize((384, 512)))
            mask, _ = get_mask_location("hd", "upper_body", model_parse, keypoints)
            mask = mask.resize((768, 1024))

            human_np = apply_exif_orientation(human_rgb.resize((384, 512)))
            human_np = convert_pil_to_numpy(human_np, format="BGR")
            detectron_device = "cuda" if device == "cuda" else "cpu"
            args = apply_net.create_argument_parser().parse_args(
                (
                    "show",
                    str(self.repo_path / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"),
                    str(self.repo_path / "ckpt" / "densepose" / "model_final_162be9.pkl"),
                    "dp_segm",
                    "-v",
                    "--opts",
                    "MODEL.DEVICE",
                    detectron_device,
                )
            )
            pose_arr = args.func(args, human_np)
            pose_arr = pose_arr[:, :, ::-1]
            pose_img = Image.fromarray(np.uint8(pose_arr)).resize((768, 1024))

            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            wear_prompt = "model is wearing " + self.garment_des
            cloth_prompt = "a photo of " + self.garment_des

            with torch.no_grad():
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        wear_prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        [cloth_prompt],
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=[negative_prompt],
                    )

                pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device=device, dtype=torch_dtype)
                cloth_tensor = tensor_transform(garment_rgb).unsqueeze(0).to(device=device, dtype=torch_dtype)

                generator = None
                if self.seed_value is not None:
                    gen_device = "cuda" if device == "cuda" else "cpu"
                    generator = torch.Generator(device=gen_device).manual_seed(int(self.seed_value))

                if device == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device=device, dtype=torch_dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(device=device, dtype=torch_dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device=device, dtype=torch_dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device=device, dtype=torch_dtype),
                            num_inference_steps=self.denoise_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_tensor,
                            text_embeds_cloth=prompt_embeds_c.to(device=device, dtype=torch_dtype),
                            cloth=cloth_tensor,
                            mask_image=mask,
                            image=human_rgb,
                            height=1024,
                            width=768,
                            ip_adapter_image=garment_rgb,
                            guidance_scale=2.0,
                        )[0]
                else:
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device=device, dtype=torch_dtype),
                        negative_prompt_embeds=negative_prompt_embeds.to(device=device, dtype=torch_dtype),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device=device, dtype=torch_dtype),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device=device, dtype=torch_dtype),
                        num_inference_steps=self.denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_tensor,
                        text_embeds_cloth=prompt_embeds_c.to(device=device, dtype=torch_dtype),
                        cloth=cloth_tensor,
                        mask_image=mask,
                        image=human_rgb,
                        height=1024,
                        width=768,
                        ip_adapter_image=garment_rgb,
                        guidance_scale=2.0,
                    )[0]

                if isinstance(images, list) and images:
                    result = images[0]
                    if isinstance(result, Image.Image):
                        return result.convert("RGB")
                if isinstance(images, Image.Image):
                    return images.convert("RGB")

            logger.warning("IDM-VTON returned unexpected output format: %s", type(images))
            return None

        except Exception as exc:
            logger.warning("IDM-VTON inference failed: %s", exc)
            import traceback
            logger.debug(traceback.format_exc())
            return None


_idm_vton_instance: Optional[IDMVTONWrapper] = None


def get_idm_vton_wrapper() -> IDMVTONWrapper:
    global _idm_vton_instance
    if _idm_vton_instance is None:
        _idm_vton_instance = IDMVTONWrapper()
    return _idm_vton_instance

