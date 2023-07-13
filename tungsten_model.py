import random
from typing import List

from tungstenkit import BaseIO, Field, Image, Option, define_model

from modules.initialize import initialize
from modules.txt2img import txt2img

class Input(BaseIO):
    prompt: str = Field(description="Input prompt")
    negative_prompt: str = Option(
        description="Specify things to not see in the output",
        default="",
    )
    image_dimensions: str = Option(
        default="512x768",
        description="Pixel dimensions of output image (width x height)",
        choices=["512x512", "512x768", "768x512"],
    )
    sampler: str = Option(
        default="Euler a",
        choices=[
            "Euler a",
            "Euler",
            "LMS",
            "Heun",
            "DPM2",
            "DPM2 a",
            "DPM++ 2S a",
            "DPM++ 2M",
            "DPM++ SDE",
            "DPM++ 2M SDE",
            "DPM fast",
            "DPM adaptive",
            "LMS Karras",
            "DPM2 Karras",
            "DPM2 a Karras",
            "DPM++ 2S a Karras",
            "DPM++ 2M Karras",
            "DPM++ SDE Karras",
            "DPM++ 2M SDE Karras",
            "DDIM",
            "PLMS",
            "UniPC",
        ],
        description="Sampler type",
    )
    samping_steps: int = Option(
        description="Number of denoising steps", ge=1, le=500, default=50
    )
    cfg_scale: float = Option(
        description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
    )
    num_outputs: int = Option(
        description="Number of output images",
        le=8,
        ge=1,
        default=1,
    )
    seed: int = Option(
        description="Random seed. Set as -1 to randomize the seed", default=-1, ge=-1
    )
    clip_skip: bool = Option(
        description="Whether to ignore the last layer of CLIP network or not",
        default=False,
    )


class Output(BaseIO):
    images: List[Image]


@define_model(
    input=Input,
    output=Output,
    batch_size=1,
    gpu_mem_gb=14,
    gpu=True,
    system_packages=["python3-opencv"],
    python_packages=[
        "torch==2.0.1",
        "torchvision==0.15.2",
        "clip @ git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1",
        "open-clip-torch @ git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b",
        "realesrgan==0.3.0",
        "pytorch-lightning==1.9.4",
        "transformers==4.25.1",
        "torchdiffeq==0.2.3",
        "torchsde==0.2.5",
        "tomesd==0.1.3",
        "kornia==0.6.12",
        "einops==0.6.1",
        "llvmlite==0.40.1",
        "numba==0.57.1",
        "basicsr==1.4.2",
        "omegaconf==2.3.0",
        "piexif==1.1.3",
        "psutil==5.9.5",
        "resize-right==0.0.2",
        "safetensors==0.3.1",
        "scikit-image==0.21.0",
        "timm==0.9.2",
        "ngrok==0.8.1",
        "fairscale==0.4.13",
        "pytz==2023.3",
        "jsonmerge==1.9.0",
        "lark==1.1.5",
        "tqdm==4.65.0",
        "clean-fid==0.1.35",
    ],
    python_version="3.10",
    include_files=[
        "configs",
        "extensions-builtin",
        "localizations",
        "models",
        "modules",
        "repositories",
    ],
)
class StableDiffusion:
    def setup(self):
        initialize()

    def predict(self, inputs: List[Input]) -> List[Output]:
        input = inputs[0]

        width, height = [int(d) for d in input.image_dimensions.split("x")]

        if input.seed == -1:
            input.seed = random.randrange(4294967294)
            print(f"Using seed {input.seed}\n")

        processed = txt2img(
            prompt=input.prompt,
            negative_prompt=input.negative_prompt,
            seed=float(input.seed),
            sampler_name=input.sampler,
            batch_size=input.num_outputs,
            steps=input.samping_steps,
            cfg_scale=input.cfg_scale,
            width=width,
            height=height,
            clip_skip=input.clip_skip,
        )
        images = [Image.from_pil_image(pil_img) for pil_img in processed.images]
        return [Output(images=images)]