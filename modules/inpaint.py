import os
import shutil
from typing import List, Optional, Tuple, Union

import numpy as np
from tungstenkit import MaskedImage, Image


def inpaint(
    masked_image: MaskedImage,
    prompt: str,
    negative_prompt: str,
    seed: float | int,
    sampler_name: str,
    batch_size: int,
    steps: int,
    denoising_strength: float,
    cfg_scale: float,
    width: int,
    height: int,
    clip_skip: int,
    loras: Optional[List[Tuple[str, float]]] = None,
    trigger_words: Optional[List[Union[str, Tuple[str, float]]]] = None,
    extra_negative_prompt_chunks: Optional[List[Union[str, Tuple[str, float]]]] = None,
    extra_positive_prompt_chunks: Optional[List[Union[str, Tuple[str, float]]]] = None,
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
    print("Full negative prompt:", negative_prompt)
    processing = StableDiffusionProcessingImg2Img(
        init_images=[masked_image.image.to_pil_image()],
        mask=masked_image.mask.to_pil_image(),
        sd_model=shared.sd_model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=float(seed),
        sampler_name=sampler_name,
        batch_size=batch_size,
        steps=steps,
        denoising_strength=denoising_strength,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        override_settings={},
    )

    # Do processing
    processed = process_images(processing)
    processing.close()

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