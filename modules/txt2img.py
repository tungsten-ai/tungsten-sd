from pathlib import Path
from typing import Optional

from modules import shared
from modules.processing import (
    Processed,
    StableDiffusionProcessingTxt2Img,
    process_images,
)


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
    lora_model: Optional[Path] = None,
) -> Processed:
    if lora_model:
        link_path = Path(shared.models_path) / "Lora" / lora_model.name
        link_path.symlink_to(lora_model)

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
        width=width,
        height=height,
        override_settings={},
    )
    processed = process_images(processing)
    processing.close()

    return processed
