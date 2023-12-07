import os
import shutil
from typing import List, Optional, Tuple, Union
from contextlib import closing

import numpy as np
from tungstenkit import Image


def img2img(
    input_image: Image,
    denoising_strength: float,
    prompt: str,
    negative_prompt: str,
    seed: float | int,
    sampler_name: str,
    batch_size: int,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    clip_skip: int,
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
    adtailer_denoising_strength: float = 0.4,
    enhance_face_with_adtailer: bool = False,
    enhance_hands_with_adtailer: bool = False,
) -> List[Image]:
    from modules import images, prompt_utils, scripts, shared
    from modules.processing import StableDiffusionProcessingImg2Img, process_images

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
    shared.opts.set("CLIP_stop_at_last_layers", clip_skip)
    print("Full prompt:", prompt)
    print()
    print("Full negative prompt:", negative_prompt)
    processing = StableDiffusionProcessingImg2Img(
        init_images=[input_image.image.to_pil_image()],
        denoising_strength=denoising_strength,
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

    processing.scripts = scripts.scripts_txt2img
    processing.script_args = [0]

    processing.scripts.scripts[0].args_from = 1
    processing.scripts.scripts[0].args_to = 3
    processing.scripts.scripts[1].args_from = 3
    processing.scripts.scripts[1].args_to = 6
    
    processing.script_args.append({
        'ad_model': 'face_yolov8s.pt' if enhance_face_with_adtailer else 'None', 
        'ad_prompt': '', 
        'ad_negative_prompt': '', 
        'ad_confidence': 0.3, 
        'ad_mask_k_largest': 0, 
        'ad_mask_min_ratio': 0, 
        'ad_mask_max_ratio': 1, 
        'ad_x_offset': 0, 
        'ad_y_offset': 0, 
        'ad_dilate_erode': 4, 
        'ad_mask_merge_invert': 'None', 
        'ad_mask_blur': 4, 
        'ad_denoising_strength': adtailer_denoising_strength, 
        'ad_inpaint_only_masked': True, 
        'ad_inpaint_only_masked_padding': 32, 
        'ad_use_inpaint_width_height': False, 
        'ad_inpaint_width': 512, 
        'ad_inpaint_height': 512, 
        'ad_use_steps': False, 
        'ad_steps': 28, 
        'ad_use_cfg_scale': False, 
        'ad_cfg_scale': 7, 
        'ad_use_checkpoint': False, 
        'ad_checkpoint': 'Use same checkpoint', 
        'ad_use_vae': False, 
        'ad_vae': 'Use same VAE', 
        'ad_use_sampler': False, 
        'ad_sampler': 'DPM++ 2M Karras', 
        'ad_use_noise_multiplier': False, 
        'ad_noise_multiplier': 1, 
        'ad_use_clip_skip': False, 
        'ad_clip_skip': 1, 
        'ad_restore_face': False, 
        'ad_controlnet_model': 'None', 
        'ad_controlnet_module': 'inpaint_global_harmonious', 
        'ad_controlnet_weight': 1, 
        'ad_controlnet_guidance_start': 0, 
        'ad_controlnet_guidance_end': 1, 
        'is_api': ()
    })

    processing.script_args.append({
        'ad_model': 'hand_yolov8n.pt' if enhance_hands_with_adtailer else 'None', 
        'ad_prompt': '', 
        'ad_negative_prompt': '', 
        'ad_confidence': 0.3, 
        'ad_mask_k_largest': 0, 
        'ad_mask_min_ratio': 0, 
        'ad_mask_max_ratio': 1, 
        'ad_x_offset': 0, 
        'ad_y_offset': 0, 
        'ad_dilate_erode': 4, 
        'ad_mask_merge_invert': 'None', 
        'ad_mask_blur': 4, 
        'ad_denoising_strength': adtailer_denoising_strength, 
        'ad_inpaint_only_masked': True, 
        'ad_inpaint_only_masked_padding': 32, 
        'ad_use_inpaint_width_height': False, 
        'ad_inpaint_width': 512, 
        'ad_inpaint_height': 512, 
        'ad_use_steps': False, 
        'ad_steps': 28, 
        'ad_use_cfg_scale': False, 
        'ad_cfg_scale': 7, 
        'ad_use_checkpoint': False, 
        'ad_checkpoint': 'Use same checkpoint', 
        'ad_use_vae': False, 
        'ad_vae': 'Use same VAE', 
        'ad_use_sampler': False, 
        'ad_sampler': 'DPM++ 2M Karras', 
        'ad_use_noise_multiplier': False, 
        'ad_noise_multiplier': 1, 
        'ad_use_clip_skip': False, 
        'ad_clip_skip': 1, 
        'ad_restore_face': False, 
        'ad_controlnet_model': 'None', 
        'ad_controlnet_module': 'inpaint_global_harmonious', 
        'ad_controlnet_weight': 1, 
        'ad_controlnet_guidance_start': 0, 
        'ad_controlnet_guidance_end': 1, 
        'is_api': ()
    })

    # Controlnet script
    processing.script_args.append(
        {
            "enabled": controlnet_pose_image is not None,
            "module": "openpose",
            "model": "controlnetxlCNXL_tencentarcOpenpose"
            if shared.sd_model.is_sdxl
            else "controlnet11Models_openpose",
            "weight": controlnet_pose_weight,
            "image": {
                "image": np.array(
                    controlnet_pose_image.to_pil_image("RGB")
                ).astype("uint8") if controlnet_pose_image is not None else None,
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
    processing.script_args.append(
        {
            "enabled": controlnet_depth_image is not None,
            "module": "depth",
            "model": "controlnetxlCNXL_tencentarcDepthMidas"
            if shared.sd_model.is_sdxl
            else "controlnet11Models_depth",
            "weight": controlnet_depth_weight,
            "image": {
                "image": np.array(
                    controlnet_depth_image.to_pil_image("RGB")
                ).astype("uint8") if controlnet_depth_image else None,
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
    processing.script_args.append(
        {
            "enabled": controlnet_reference_only_image is not None,
            "module": "reference_only",
            "model": None,
            "weight": controlnet_reference_only_weight,
            "image": {
                "image": np.array(
                    controlnet_reference_only_image.to_pil_image("RGB")
                ).astype("uint8") if controlnet_reference_only_image is not None else None,
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
    with closing(processing):
        processed = scripts.scripts_txt2img.run(processing, *processing.script_args)
        if processed is None:
            processed = process_images(processing)

    output_pil_images = processed.images[:batch_size]

    # Save images with metadata
    ret = []
    if os.path.exists(".results"):
        shutil.rmtree(".results")
    os.mkdir(".results")
    for i, pil_image in enumerate(output_pil_images):
        saved_path, _ = images.save_image(
            pil_image,
            os.path.abspath(".results"),
            "",
            processing.seeds[i],
            processing.prompts[i],
            info=processed.infotexts[i],
            p=processing,
        )
        ret.append(Image.from_path(saved_path))

    return ret
