import os
import random
import shutil
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tungstenkit import BaseIO, Binary, Field, Image, Option, define_model

from check_if_sdxl import check_if_sdxl
from modules.initialize import initialize, initialize_vae, load_vae_weights
from modules.txt2img import txt2img

VAE_FILE_PATHS = glob("models/VAE/*")
MODEL_FILES = glob("models/Stable-diffusion/*.safetensors")
assert len(MODEL_FILES) > 0, "Stable diffusion checkpoint not found"
IS_SDXL = check_if_sdxl(MODEL_FILES[0])

SAMPLERS = [
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
    "Restart",
]
SD_VAES_IN_BASE_IMAGE = [
    "vae-ft-mse-840000-ema-pruned_fp16.safetensors",
    "orangemix.vae.pt",
    "kl-f8-anime2_fp16.safetensors",
    "anything_fp16.safetensors",
    "blessed2_fp16.safetensors",
    "clearvae_v2.3_fp16.safetensors",
]
SDXL_VAES_IN_BASE_IMAGE = [
    "sdxl_vae.safetensors",
]
ALL_VAE_FILE_PATHS = VAE_FILE_PATHS + [
    "models/VAE/" + vae_name
    for vae_name in (SDXL_VAES_IN_BASE_IMAGE if IS_SDXL else SD_VAES_IN_BASE_IMAGE)
    if vae_name not in [p.split("/")[-1] for p in VAE_FILE_PATHS]
]

DEFAULT_SAMPLER = "Restart"


class Input(BaseIO):
    prompt: str = Field(description="Input prompt")
    negative_prompt: str = Option(
        description="Specify things to not see in the output",
        default="",
    )
    width: int = Option(
        description="Output image width",
        default=768 if IS_SDXL else 512,
        ge=512,
        le=2048 if IS_SDXL else 1024,
    )
    height: int = Option(
        description="Output image height",
        default=1344 if IS_SDXL else 768,
        ge=512,
        le=2048 if IS_SDXL else 1024,
    )
    num_outputs: int = Option(
        description="Number of output images",
        le=8,
        ge=1,
        default=1,
    )
    seed: int = Option(
        description="Random seed. Set as -1 to randomize output.",
        default=-1,
        ge=-1,
        le=4294967293,
    )
    reference_image: Optional[Image] = Option(
        description="Image that the output should be similar to",
        default=None,
    )
    reference_image_strength: float = Option(
        description="Strength of applying reference_image. Used only when reference_image is given.",
        default=1.0,
        ge=0.0,
        le=2.0,
    )
    reference_pose_image: Optional[Image] = Option(
        description="Image with a reference pose",
        default=None,
    )
    reference_pose_strength: float = Option(
        description="Strength of applying reference_pose_image. Used only when reference_pose_image is given.",  # noqa: E501
        default=1.0,
        ge=0.0,
        le=2.0,
    )
    reference_depth_image: Optional[Image] = Option(
        description="Image with a reference depth",
        default=None,
    )
    reference_depth_strength: float = Option(
        description="Strength of applying reference_depth_image. Used only when reference_depth_image is given.",  # noqa: E501
        default=1.0,
        ge=0.0,
        le=2.0,
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
    clip_skip: bool = Option(
        description="Whether to ignore the last layer of CLIP network or not",
        default=not IS_SDXL,
    )
    vae: str = Option(
        description="Select VAE",
        default=VAE_FILE_PATHS[0].split("/")[-1] if VAE_FILE_PATHS else "None",
        choices=["None"] + [vae_path.split("/")[-1] for vae_path in ALL_VAE_FILE_PATHS],
    )
    lora_1: Optional[Binary] = Option(
        description="LoRA file. Apply by writing the following in prompt: <lora:[FILE_NAME_WITHOUT_EXTENSION]:[MAGNITUDE]>",  # noqa: E501
        default=None,
    )
    lora_2: Optional[Binary] = Option(
        description="LoRA file. Apply by writing the following in prompt: <lora:[FILE_NAME_WITHOUT_EXTENSION]:[MAGNITUDE]>",  # noqa: E501
        default=None,
    )
    lora_3: Optional[Binary] = Option(
        description="LoRA file. Apply by writing the following in prompt: <lora:[FILE_NAME_WITHOUT_EXTENSION]:[MAGNITUDE]>",  # noqa: E501
        default=None,
    )
    embedding_1: Optional[Binary] = Option(
        description="Embedding file (textural inversion). Apply by writing the following in prompt or negative prompt: ([FILE_NAME_WITHOUT_EXTENSION]:[MAGNITUDE])",  # noqa: E501
        default=None,
    )
    embedding_2: Optional[Binary] = Option(
        description="Embedding file (textural inversion). Apply by writing the following in prompt or negative prompt: ([FILE_NAME_WITHOUT_EXTENSION]:[MAGNITUDE])",  # noqa: E501
        default=None,
    )
    embedding_3: Optional[Binary] = Option(
        description="Embedding file (textural inversion). Apply by writing the following in prompt or negative prompt: ([FILE_NAME_WITHOUT_EXTENSION]:[MAGNITUDE])",  # noqa: E501
        default=None,
    )
    disable_prompt_modification: bool = Option(
        description="Disable automatically adding suggested prompt modification. Built-in LoRAs and trigger words will remain.",  # noqa: E501
        default=False,
    )


class Output(BaseIO):
    images: List[Image]


@define_model(
    input=Input,
    output=Output,
    batch_size=1,
    gpu=True,
    gpu_mem_gb=14,
    include_files=[
        "configs",
        "extensions-builtin",
        "extensions/sd-webui-controlnet",
        "localizations",
        "models/Stable-diffusion",
        "models/VAE",
        "models/Lora",
        "modules",
        "repositories",
        "embeddings",
        "check_if_sdxl.py",
    ],
    base_image="mjpyeon/tungsten-sd-base:v2",
)
class StableDiffusion:
    def setup(self):
        initialize(
            is_sdxl=IS_SDXL,
            default_sampler=DEFAULT_SAMPLER,
        )
        initialize_vae()
        dummy_input = Input(
            prompt="dummy",
            samping_steps=1,
        )
        self.predict([dummy_input])

    def predict(self, inputs: List[Input]) -> List[Output]:
        input = inputs[0]

        # Put extra loras and embeddings to its directory
        loras: List[Path] = []
        embeddings: List[Path] = []
        try:
            _prepare_loras_and_embeddings(input, loras, embeddings)

            # Assign random seed
            if input.seed == -1:
                input.seed = random.randrange(4294967294)
                print(f"Using seed {input.seed}\n")

            load_vae_weights(
                os.path.join("models", "VAE", input.vae)
                if input.vae != "None"
                else None
            )
            try:
                # Generate image
                images = txt2img(
                    prompt=input.prompt,
                    negative_prompt=input.negative_prompt,
                    seed=float(input.seed),
                    sampler_name=input.sampler,
                    batch_size=input.num_outputs,
                    steps=input.samping_steps,
                    cfg_scale=input.cfg_scale,
                    width=input.width,
                    height=input.height,
                    clip_skip=input.clip_skip,
                    loras=self.get_loras(input),
                    trigger_words=self.get_trigger_words(input),
                    extra_positive_prompt_chunks=[]
                    if input.disable_prompt_modification
                    else self.get_extra_prompt_chunks(input),
                    extra_negative_prompt_chunks=[]
                    if input.disable_prompt_modification
                    else self.get_extra_negative_prompt_chunks(input),
                    controlnet_pose_image=input.reference_pose_image,
                    controlnet_depth_image=input.reference_depth_image,
                    controlnet_reference_only_image=input.reference_image,
                    controlnet_pose_weight=input.reference_pose_strength,
                    controlnet_depth_weight=input.reference_depth_strength,
                    controlnet_reference_only_weight=input.reference_image_strength,
                )

                return [Output(images=images)]
            finally:
                initialize_vae()

        finally:
            _cleanup_loras_and_embeddings(loras, embeddings)

    def get_loras(self, input: Input) -> List[Tuple[str, float]]:
        """
        Declare LoRAs to use in the format of (LORA_FILE_NAME, WEIGHT).

        The LoRA weight file named LORA_FILE_NAME should exist in `models/LoRA` directory.

        Examples:
          - `[("add_detail", 0.5)]` -> Put `<lora:add_detail:0.5>` at the end of the prompt.
          - `[("add_detail", input.detail)]` -> Put `<lora:add_detail:{detail field in input}>` at the end of the prompt. # noqa: E501
        """

        return []

    def get_trigger_words(self, input: Input) -> List[Union[str, Tuple[str, float]]]:
        """
        Declare trigger words to be inserted at the start of the prompt.

        Examples:
          - `["trigger1"]` -> Put `<lora:add_detail:0.5>` at the start of the prompt.
          - `[("trigger2", input.magnitude)]` -> Put `(trigger2:{magnitude field in input})` at the start of the prompt. # noqa: E501
        """

        return []

    def get_extra_prompt_chunks(
        self, input: Input
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Declare default prompt chunks.

        Using this, you can use textual inversion.

        Examples
          - `["hello"]` -> Put `hello` to the prompt.
          - `[("hello", 1.1)]` -> Put `(hello:1.1)` at the end of the prompt.
        """

        return []

    def get_extra_negative_prompt_chunks(
        self, input: Input
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Declare default negative prompt chunks.

        Using this, you can use textual inversion.

        Examples
          - `["hello"]` -> Put `hello` to the negative prompt (w/ whitespace if required).
          - `[("hello", 1.1), "world"]` -> Put `(hello:1.1), world` to the negative prompt.
        """
        return []


def _prepare_loras_and_embeddings(
    input: Input, loras_list: List[Path], embeddings_list: List[Path]
):
    loras_dir_path = Path("models/Lora")
    embeddings_dir_path = Path("embeddings")

    loras_list.extend(
        [
            lora.path
            for lora in [
                getattr(input, field_name)
                for field_name in input.__fields__.keys()
                if field_name.startswith("lora_")
            ]
            if lora is not None and not (loras_dir_path / lora.path.parts[-1]).exists()
        ]
    )
    embeddings_list.extend(
        [
            embedding.path
            for embedding in [
                getattr(input, field_name)
                for field_name in input.__fields__.keys()
                if field_name.startswith("embedding_")
            ]
            if embedding is not None
            and not (embeddings_dir_path / embedding.path.parts[-1]).exists()
        ]
    )

    for lora_path in loras_list:
        shutil.move(lora_path, loras_dir_path)
    for embedding_path in embeddings_list:
        shutil.move(embedding_path, embeddings_dir_path)


def _cleanup_loras_and_embeddings(loras_list: List[Path], embeddings_list: List[Path]):
    loras_dir_path = Path("models/Lora")
    embeddings_dir_path = Path("embeddings")
    for lora_path in loras_list:
        if (loras_dir_path / lora_path.parts[-1]).exists():
            os.remove(loras_dir_path / lora_path.parts[-1])
    for embedding_path in embeddings_list:
        if (embeddings_dir_path / embedding_path.parts[-1]).exists():
            os.remove(embeddings_dir_path / embedding_path.parts[-1])
