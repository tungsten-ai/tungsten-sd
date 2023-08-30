import random
from glob import glob
from typing import List, Optional, Tuple, Union

from tungstenkit import BaseIO, Field, Image, Option, define_model

from check_if_sdxl import check_if_sdxl
from modules.initialize import initialize
from modules.txt2img import txt2img

VAE_FILE_PATHS = glob("models/VAE/*")
MODEL_FILES = glob("models/Stable-diffusion/*.safetensors")
assert len(MODEL_FILES) > 0, "Stable diffusion checkpoint not found"
IS_SDXL = check_if_sdxl(MODEL_FILES[0])


SD_OUTPUT_DIMS = [
    "512x512",
    "512x768",
    "768x512",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "1536x1536",
    "1536x2304",
    "2304x1536",
    "2048x2048",
    "2048x3072",
    "3072x2048",
]
SDXL_OUTPUT_DIMS = [
    "1024x1024",
    "1152x896",
    "896x1152",
    "1216x832",
    "832x1216",
    "1344x768",
    "768x1344",
    "1536x640",
    "640x1536",
    "2048x2048",
    "2304x1792",
    "1792x2304",
    "2432x1664",
    "1664x2432",
    "2688x1536",
    "1536x2688",
    "3072x1280",
    "1280x3072",
]


class BaseSDInput(BaseIO):
    prompt: str = Field(description="Input prompt")

    negative_prompt: str = Option(
        description="Specify things to not see in the output",
        default="",
    )
    image_dimensions: str = Option(
        default=SDXL_OUTPUT_DIMS[0] if IS_SDXL else SD_OUTPUT_DIMS[0],
        description="Pixel dimensions of output image (width x height)",
        choices=SDXL_OUTPUT_DIMS if IS_SDXL else SD_OUTPUT_DIMS,
    )
    num_outputs: int = Option(
        description="Number of output images",
        le=8,
        ge=1,
        default=1,
    )
    seed: int = Option(
        description="Random seed. Set as -1 to randomize the seed",
        default=-1,
        ge=-1,
        le=4294967293,
    )
    sampler: str = Option(
        default="DPM++ SDE Karras",
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
        description="Number of denoising steps", ge=1, le=100, default=20
    )
    cfg_scale: float = Option(
        description="Scale for classifier-free guidance", ge=1, le=20, default=7
    )
    clip_skip: bool = Option(
        description="Whether to ignore the last layer of CLIP network or not",
        default=True,
    )


class SDInput(BaseSDInput):
    reference_image: Optional[Image] = Option(
        description="Image with a reference architecture shape",
        default=None,
    )


SDXLInput = SDInput


class Output(BaseIO):
    images: List[Image]


@define_model(
    input=SDXLInput if IS_SDXL else SDInput,
    output=Output,
    batch_size=1,
    gpu=True,
    gpu_mem_gb=14,
    system_packages=["python3-opencv"],
    python_packages=[
        "torch==2.0.1",
        "torchvision==0.15.2",
        "clip @ git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1",
        "open-clip-torch @ git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b",  # noqa: E501
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
        "xformers==0.0.21",
    ],
    python_version="3.10",
    include_files=[
        "configs",
        "extensions-builtin",
        "extensions",
        "localizations",
        "models",
        "modules",
        "repositories",
        "embeddings",
        "check_if_sdxl.py",
    ],
)
class StableDiffusion:
    def setup(self):
        initialize(
            vae_file_path=VAE_FILE_PATHS[0] if VAE_FILE_PATHS else None, is_sdxl=IS_SDXL
        )

    def predict(self, inputs: List[BaseSDInput]) -> List[Output]:
        input = inputs[0]

        # Output image size
        width, height = [int(d) for d in input.image_dimensions.split("x")]

        # Assign random seed
        if input.seed == -1:
            input.seed = random.randrange(4294967294)
            print(f"Using seed {input.seed}\n")

        # Generate image
        images = txt2img(
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
            loras=self.get_loras(input),
            default_positive_prompt_chunks=self.get_extra_prompt_chunks(input),
            default_negative_prompt_chunks=self.get_extra_negative_prompt_chunks(input),
            controlnet_pose_image=None if IS_SDXL else input.reference_image,
            controlnet_depth_image=None if IS_SDXL else input.reference_image,
        )

        return [Output(images=images)]

    def get_loras(self, input: BaseSDInput) -> List[Tuple[str, float]]:
        """
        Declare LoRAs to use in the format of (LORA_FILE_NAME, WEIGHT).

        The LoRA weight file named LORA_FILE_NAME should exist in `models/LoRA` directory.

        Examples:
          - `[("add_detail", 0.5)]` -> Put `<lora:add_detail:0.5>` to the prompt.
          - `[("add_detail", input.detail)]` -> Put `<lora:add_detail:{detail field in input}>` to the prompt. # noqa: E501
        """

        return []

    def get_extra_prompt_chunks(
        self, input: BaseSDInput
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Declare default prompt chunks.

        Using this, you can use textual inversion.

        Examples
          - `["hello"]` -> Put `hello` to the prompt (w/ whitespace if required).
          - `[("hello", 1.1), "world"]` -> Put `(hello:1.1), world` to the prompt.
        """

        return []

    def get_extra_negative_prompt_chunks(
        self, input: BaseSDInput
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Declare default negative prompt chunks.

        Using this, you can use textual inversion.

        Examples
          - `["hello"]` -> Put `hello` to the negative prompt (w/ whitespace if required).
          - `[("hello", 1.1), "world"]` -> Put `(hello:1.1), world` to the negative prompt.
        """
        return []
