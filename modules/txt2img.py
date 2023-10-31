import itertools
from typing import List, Optional, Tuple, Union

import numpy as np
from tungstenkit import Image


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
    controlnet_reference_only_image: Optional[Image] = None,
    controlnet_pose_weight: float = 1.0,
    controlnet_depth_weight: float = 1.0,
    controlnet_reference_only_weight: float = 1.0,
    loras: Optional[List[Tuple[str, float]]] = None,
    trigger_words: Optional[List[Union[str, Tuple[str, float]]]] = None,
    extra_negative_prompt_chunks: Optional[List[Union[str, Tuple[str, float]]]] = None,
    extra_positive_prompt_chunks: Optional[List[Union[str, Tuple[str, float]]]] = None,
) -> List[Image]:
    from modules import prompt_utils, scripts, shared
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images

    controlnet_counts = sum(
        cn is not None
        for cn in [
            controlnet_pose_image,
            controlnet_depth_image,
            controlnet_reference_only_image,
        ]
    )
    use_controlnet = controlnet_counts > 0

    # Modify positive/negative prompts
    if extra_negative_prompt_chunks:
        for embedding in extra_negative_prompt_chunks:
            negative_prompt = prompt_utils.suppress_plain_keyword(
                embedding if isinstance(embedding, str) else embedding[0],
                negative_prompt,
            )
        for embedding in extra_negative_prompt_chunks:
            negative_prompt_increment = (
                embedding
                if isinstance(embedding, str)
                else f"({embedding[0]}:{embedding[1]})"
            )
            if negative_prompt:
                negative_prompt += ", " + negative_prompt_increment
            else:
                negative_prompt = negative_prompt_increment

    if trigger_words:
        for word in trigger_words:
            prompt = prompt_utils.suppress_plain_keyword(
                word if isinstance(word, str) else word[0], prompt
            )
            prompt = (
                f"{word}, " if isinstance(word, str) else f"({word[0]}:{word[1]}), "
            ) + prompt

    if extra_positive_prompt_chunks:
        for embedding in extra_positive_prompt_chunks:
            prompt = prompt_utils.suppress_plain_keyword(
                embedding if isinstance(embedding, str) else embedding[0], prompt
            )
        for embedding in extra_positive_prompt_chunks:
            prompt += (
                f", {embedding}"
                if isinstance(embedding, str)
                else f", ({embedding[0]}:{embedding[1]})"
            )

    if loras:
        for lora_keyword, _ in loras:
            prompt = prompt_utils.suppress_lora_keyword(lora_keyword, prompt)
        for lora_keyword, lora_weight in loras:
            if lora_weight != 0.0:
                prompt += f", <lora:{lora_keyword}:{lora_weight}>"

    # Prepare processing
    shared.opts.set("CLIP_stop_at_last_layers", 2 if clip_skip else 1)
    print("Full prompt:", prompt)
    print("Full negative prompt:", negative_prompt)
    processing = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=float(seed),
        sampler_name=sampler_name,
        batch_size=batch_size,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
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
                    "model": "controlnetxlCNXL_tencentarcOpenpose"
                    if shared.sd_model.is_sdxl
                    else "controlnet11Models_openpose",
                    "weight": controlnet_pose_weight,
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
                    "model": "controlnetxlCNXL_tencentarcDepthMidas"
                    if shared.sd_model.is_sdxl
                    else "controlnet11Models_depth",
                    "weight": controlnet_depth_weight,
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
        if controlnet_reference_only_image:
            processing.script_args.append(
                {
                    "enabled": True,
                    "module": "reference_only",
                    "model": None,
                    "weight": controlnet_reference_only_weight,
                    "image": {
                        "image": np.array(
                            controlnet_reference_only_image.to_pil_image("RGB")
                        ).astype("uint8"),
                        "mask": None,
                    },
                    "resize_mode": "Crop and Resize",
                    "lowvram": False,
                    "processor_res": -1,
                    "threshold_a": 0.5,
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

    output_pil_images = processed.images[:batch_size]

    images = [Image.from_pil_image(pil_img) for pil_img in output_pil_images]

    return images
