import os
import random
import shutil
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tungstenkit import BaseIO, Binary, Field, Image, Option, define_model

from check_if_sdxl import check_if_sdxl
from modules.initialize import initialize
from modules.txt2gif import txt2gif

ANIMATEDIFF_FILE_PATHS = glob("extensions/sd-webui-animatediff/model/*")
VAE_FILE_PATHS = glob("models/VAE/*")
MODEL_FILES = glob("models/Stable-diffusion/*.safetensors")
assert len(MODEL_FILES) > 0, "Stable diffusion checkpoint not found"
IS_SDXL = check_if_sdxl(MODEL_FILES[0])

SD_OUTPUT_DIMS = [
    "512x512",
    "512x768",
    "768x512",
]
SDXL_OUTPUT_DIMS = [
    "1024x1024",
    "1168x880",
    "896x1152",
    "1216x832",
    "832x1216",
    "1280x768",
    "768x1280",
    "1344x768",
    "768x1344",
    "1536x640",
    "640x1536",
]
SAMPLERS = [
    "Restart",
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DPM++ 2M SDE Exponential",
    "DPM++ 2M SDE Karras",
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
    "DPM++ 2M SDE Heun",
    "DPM++ 2M SDE Heun Karras",
    "DPM++ 2M SDE Heun Exponential",
    "DPM++ 3M SDE",
    "DPM++ 3M SDE Karras",
    "DPM++ 3M SDE Exponential",
    "DPM fast",
    "DPM adaptive",
    "LMS Karras",
    "DPM2 Karras",
    "DPM2 a Karras",
    "DPM++ 2S a Karras",
]
DEFAULT_SAMPLER = "DPM++ SDE Karras"


class BaseInput(BaseIO):
    prompt: str = Field(description="Input prompt")
    negative_prompt: str = Option(
        description="Specify things to not see in the output",
        default="",
    )
    gif_length_seconds: int = Option(
        description="GIF length in seconds",
        ge=1,
        le=5,
        default=2,
    )
    gif_fps: int = Option(
        description="GIF FPS (frames per seconds)",
        ge=1,
        le=15,
        default=8,
    )


class SDInput(BaseInput):
    # reference_image: Optional[Image] = Option(
    #     description="Image that the output should be similar to",
    #     default=None,
    # )
    # reference_pose_image: Optional[Image] = Option(
    #     description="Image with a reference pose",
    #     default=None,
    # )
    # reference_depth_image: Optional[Image] = Option(
    #     description="Image with a reference depth",
    #     default=None,
    # )
    image_dimensions: str = Option(
        default=SD_OUTPUT_DIMS[0],
        description="Pixel dimensions of output image (width x height)",
        choices=SD_OUTPUT_DIMS,
    )
    seed: int = Option(
        description="Random seed. Set as -1 to randomize the seed",
        default=-1,
        ge=-1,
        le=4294967293,
    )
    sampler: str = Option(
        default=DEFAULT_SAMPLER,
        choices=SAMPLERS,
        description="Sampler type",
    )
    samping_steps: int = Option(
        description="Number of denoising steps", ge=1, le=100, default=20
    )
    cfg_scale: float = Option(
        description="Scale for classifier-free guidance", ge=1, le=20, default=7
    )
    clip_skip: int = Option(
        description="The number of last layers of CLIP network to skip",
        default=2,
        le=11,
        ge=1,
    )
    lora: Optional[Binary] = Option(
        description="LoRA file. You can apply and adjust the magnitude by putting the following to the prompt: <lora:[FILE_NAME]:[MAGNITUDE]>",  # noqa: E501
        default=None,
    )


class SDXLInput(BaseInput):
    image_dimensions: str = Option(
        default=SDXL_OUTPUT_DIMS[0],
        description="Pixel dimensions of output image (width x height)",
        choices=SDXL_OUTPUT_DIMS,
    )
    seed: int = Option(
        description="Random seed. Set as -1 to randomize the seed",
        default=-1,
        ge=-1,
        le=4294967293,
    )
    sampler: str = Option(
        default=DEFAULT_SAMPLER,
        choices=SAMPLERS,
        description="Sampler type",
    )
    samping_steps: int = Option(
        description="Number of denoising steps", ge=1, le=100, default=20
    )
    cfg_scale: float = Option(
        description="Scale for classifier-free guidance", ge=1, le=20, default=7
    )
    clip_skip: int = Option(
        description="The number of last layers of CLIP network to skip",
        default=1,
        le=11,
        ge=1,
    )
    lora: Optional[Binary] = Option(
        description="LoRA file. You can apply and adjust the magnitude by putting the following to the prompt: <lora:[FILE_NAME]:[MAGNITUDE]>",  # noqa: E501
        default=None,
    )


class Output(BaseIO):
    gif: Image


@define_model(
    input=SDXLInput if IS_SDXL else SDInput,
    output=Output,
    batch_size=1,
    gpu=True,
    gpu_mem_gb=14,
    include_files=[
        "configs",
        "extensions-builtin",
        "extensions",
        "localizations",
        "models/Stable-diffusion",
        "models/VAE",
        "models/Lora",
        "modules",
        "repositories",
        "embeddings",
        "check_if_sdxl.py",
    ],
    exclude_files=["models/ControlNet*", "extensions/sd-webui-controlnet*"],
    base_image="mjpyeon/tungsten-sd-txt2gif-base:v1",
)
class StableDiffusionTxt2Gif:
    def setup(self):
        initialize(
            vae_file_path=VAE_FILE_PATHS[0] if VAE_FILE_PATHS else None,
            is_sdxl=IS_SDXL,
            default_sampler=DEFAULT_SAMPLER,
        )
        input_cls = SDXLInput if IS_SDXL else SDInput
        dummy_input = input_cls(
            prompt="dummy",
            samping_steps=1,
        )
        self.predict([dummy_input])

    def predict(self, inputs: List[BaseInput]) -> List[Output]:
        input = inputs[0]

        # Put extra lora to its directory
        if input.lora is not None:
            shutil.move(input.lora.path, "models/Lora")
        try:
            # Output image size
            width, height = [int(d) for d in input.image_dimensions.split("x")]

            # Assign random seed
            if input.seed == -1:
                input.seed = random.randrange(4294967294)
                print(f"Using seed {input.seed}\n")

            # Generate image
            gif = txt2gif(
                prompt=input.prompt,
                negative_prompt=input.negative_prompt,
                seed=float(input.seed),
                sampler_name=input.sampler,
                steps=input.samping_steps,
                cfg_scale=input.cfg_scale,
                width=width,
                height=height,
                model_name=Path(ANIMATEDIFF_FILE_PATHS[0]).name,
                gif_fps=input.gif_fps,
                gif_frames_count=input.gif_fps * input.gif_length_seconds,
                clip_skip=input.clip_skip,
                loras=self.get_loras(input),
                trigger_words=self.get_trigger_words(input),
                extra_positive_prompt_chunks=self.get_extra_prompt_chunks(input),
                extra_negative_prompt_chunks=self.get_extra_negative_prompt_chunks(
                    input
                ),
                # controlnet_pose_image=None if IS_SDXL else input.reference_pose_image,
                # controlnet_depth_image=None if IS_SDXL else input.reference_depth_image,
                # controlnet_reference_only_image=None
                # if IS_SDXL
                # else input.reference_image,
            )

            return [Output(gif=gif)]

        finally:
            if input.lora is not None:
                os.remove(Path("models/Lora") / input.lora.path.parts[-1])

    def get_loras(self, input: BaseInput) -> List[Tuple[str, float]]:
        """
        Declare LoRAs to use in the format of (LORA_FILE_NAME, WEIGHT).

        The LoRA weight file named LORA_FILE_NAME should exist in `models/LoRA` directory.

        Examples:
          - `[("add_detail", 0.5)]` -> Put `<lora:add_detail:0.5>` to the prompt.
          - `[("add_detail", input.detail)]` -> Put `<lora:add_detail:{detail field in input}>` to the prompt. # noqa: E501
        """

        return []

    def get_trigger_words(self, input: BaseInput) -> List[str]:
        """
        Declare trigger words to be inserted at the start of the prompt.
        """

        return []

    def get_extra_prompt_chunks(
        self, input: BaseInput
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
        self, input: BaseInput
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Declare default negative prompt chunks.

        Using this, you can use textual inversion.

        Examples
          - `["hello"]` -> Put `hello` to the negative prompt (w/ whitespace if required).
          - `[("hello", 1.1), "world"]` -> Put `(hello:1.1), world` to the negative prompt.
        """
        return ["epicnegative"]
        return []


if __name__ == "__main__":
    model = StableDiffusionTxt2Gif()
    model.setup()
    outputs = model.predict([SDInput(prompt="woman", seed=342192174)])
    print(outputs[0].gif.path)
