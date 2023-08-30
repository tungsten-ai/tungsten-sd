import itertools
from typing import List, Optional, Tuple, Union

import numpy as np
from tungstenkit import Image

from modules import prompt_utils, scripts, shared
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.realesrgan_model import UpscalerRealESRGAN

GENERATED_IMG_SIZES = [512, 768]
SCALE_FACTORS = range(1, 5)
AVAILABLE_IMG_SIZES = [
    size * scale_factor
    for size in GENERATED_IMG_SIZES
    for scale_factor in SCALE_FACTORS
]

upscaler = None


def txt2img(
    prompt: str,
    negative_prompt: str,
    seed: float | int,
    sampler_name: str,
    batch_size: int,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    clip_skip: bool,
    controlnet_pose_image: Optional[Image] = None,
    controlnet_depth_image: Optional[Image] = None,
    loras: Optional[List[Tuple[str, float]]] = None,
    default_negative_prompt_chunks: Optional[
        List[Union[str, Tuple[str, float]]]
    ] = None,
    default_positive_prompt_chunks: Optional[
        List[Union[str, Tuple[str, float]]]
    ] = None,
) -> List[Image]:
    global upscaler

    use_controlnet = bool(controlnet_pose_image or controlnet_depth_image)

    # Modify prompt
    if loras:
        for lora_keyword, _ in loras:
            prompt = prompt_utils.suppress_lora_keyword(lora_keyword, prompt)
        for lora_keyword, lora_weight in loras:
            prompt += f", <lora:{lora_keyword}:{lora_weight}>"

    if default_positive_prompt_chunks:
        for embedding in default_positive_prompt_chunks:
            prompt = prompt_utils.suppress_plain_keyword(
                embedding if isinstance(embedding, str) else embedding[0], prompt
            )
        for embedding in default_positive_prompt_chunks:
            prompt += (
                f", {embedding}"
                if isinstance(embedding, str)
                else f", ({embedding[0]}:{embedding[1]})"
            )

    if default_negative_prompt_chunks:
        for embedding in default_negative_prompt_chunks:
            negative_prompt = prompt_utils.suppress_plain_keyword(
                embedding if isinstance(embedding, str) else embedding[0],
                negative_prompt,
            )
        for embedding in default_negative_prompt_chunks:
            negative_prompt_increment = (
                embedding
                if isinstance(embedding, str)
                else f"({embedding[0]}:{embedding[1]})"
            )
            if negative_prompt:
                negative_prompt += ", " + negative_prompt_increment
            else:
                negative_prompt = negative_prompt_increment

    # Upscaler config
    (gen_width, gen_height), scale_factor = _get_generated_image_size_and_scale_factor(
        width, height
    )

    # Prepare processing
    shared.opts.set("CLIP_stop_at_last_layers", 2 if clip_skip else 1)
    processing = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=float(seed),
        sampler_name=sampler_name,
        batch_size=batch_size,
        steps=steps,
        cfg_scale=cfg_scale,
        width=gen_width,
        height=gen_height,
        override_settings={},
    )

    # Controlnet script
    if use_controlnet:
        processing.scripts = scripts.scripts_txt2img
        processing.script_args = []
        if controlnet_pose_image:
            processing.script_args.append(
                {
                    "enabled": True,
                    "module": "openpose",
                    "model": "controlnet11Models_openpose",
                    "weight": 0.5 if controlnet_depth_image else 1.0,
                    "image": {
                        "image": np.array(
                            controlnet_pose_image.to_pil_image("RGB")
                        ).astype("uint8"),
                        "mask": None,
                    },
                    "resize_mode": "Crop and Resize",
                    "lowvram": False,
                    "processor_res": 512,
                    "threshold_a": -1,
                    "threshold_b": -1,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "control_mode": "Balanced",
                    "pixel_perfect": False,
                    "input_mode": "simple",
                    "batch_images": "",
                    "output_dir": "",
                    "loopback": False,
                }
            )
        if controlnet_depth_image:
            processing.script_args.append(
                {
                    "enabled": True,
                    "module": "depth",
                    "model": "controlnet11Models_depth",
                    "weight": 0.5 if controlnet_pose_image else 1.0,
                    "image": {
                        "image": np.array(
                            controlnet_depth_image.to_pil_image("RGB")
                        ).astype("uint8"),
                        "mask": None,
                    },
                    "resize_mode": "Crop and Resize",
                    "lowvram": False,
                    "processor_res": 512,
                    "threshold_a": -1,
                    "threshold_b": -1,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "control_mode": "Balanced",
                    "pixel_perfect": False,
                    "input_mode": "simple",
                    "batch_images": "",
                    "output_dir": "",
                    "loopback": False,
                }
            )

    # Do processing
    processed = process_images(processing)

    processing.close()

    generated_pil_images = processed.images
    if controlnet_pose_image:
        generated_pil_images = generated_pil_images[:-1]
    if controlnet_depth_image:
        generated_pil_images = generated_pil_images[:-1]

    # Upscale
    if scale_factor > 1:
        print(
            f"\nUpscale generated images: {gen_width}x{gen_height} "
            f"-> {width}x{height}"
        )
        output_pil_images = []
        if upscaler is None:
            upscaler = UpscalerRealESRGAN("models/RealESRGAN")

        # Assume that there is only one upscaler (modules/realesrgan_model.py)
        upscale_info = upscaler.scalers[0]
        upscale_info.scale = scale_factor
        path = upscale_info.data_path

        # Run sequentially due to memory issue.
        for i, img in enumerate(generated_pil_images):
            print(f"Upscale image {i+1}...")
            output_pil_images.append(upscaler.do_upscale(img.convert("RGB"), path))

        print("Done.")
    else:
        output_pil_images = generated_pil_images

    images = [Image.from_pil_image(pil_img) for pil_img in output_pil_images]

    return images


def _get_generated_image_size_and_scale_factor(desired_width: int, desired_height: int):
    assert (
        desired_height in AVAILABLE_IMG_SIZES
    ), f"Invalid image height. Available heights: {AVAILABLE_IMG_SIZES}"
    assert (
        desired_width in AVAILABLE_IMG_SIZES
    ), f"Invalid image width. Available widths: {AVAILABLE_IMG_SIZES}"
    scale_factor, width, height = 1, desired_width, desired_height
    if (
        desired_width not in GENERATED_IMG_SIZES
        or desired_height not in GENERATED_IMG_SIZES
    ):
        for _width, _height in itertools.product(
            GENERATED_IMG_SIZES, GENERATED_IMG_SIZES
        ):
            if (
                desired_width % _width == 0
                and desired_height % _height == 0
                and desired_width // _width == desired_height // _height
            ):
                scale_factor = desired_width // _width
                width = _width
                height = _height
                break
    return (width, height), scale_factor
