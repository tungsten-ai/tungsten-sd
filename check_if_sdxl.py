import json
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sd_configs_path = os.path.join(script_path, "configs")

sd_path = os.path.join(script_path, "repositories/stable-diffusion-stability-ai")
sd_default_config = os.path.join(sd_configs_path, "v1-inference.yaml")
path_dirs = [
    (sd_path, "ldm", "Stable Diffusion", []),
    (
        os.path.join(sd_path, "../generative-models"),
        "sgm",
        "Stable Diffusion XL",
        ["sgm"],
    ),
    (
        os.path.join(sd_path, "../k-diffusion"),
        "k_diffusion/sampling.py",
        "k_diffusion",
        ["atstart"],
    ),
    (
        os.path.join(sd_path, "../CodeFormer"),
        "inference_codeformer.py",
        "CodeFormer",
        [],
    ),
]

paths = {}
for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(
            f"Warning: {what} not found at path {must_exist_path}",
            file=sys.stderr,
        )
    else:
        d = os.path.abspath(d)
        paths[what] = d


sd_configs_path = os.path.join(script_path, "configs")

sd_configs_path = sd_configs_path
sd_repo_configs_path = os.path.join(
    paths["Stable Diffusion"], "configs", "stable-diffusion"
)
sd_xl_repo_configs_path = os.path.join(
    paths["Stable Diffusion XL"], "configs", "inference"
)


config_default = sd_default_config
config_sd2 = os.path.join(sd_repo_configs_path, "v2-inference.yaml")
config_sd2v = os.path.join(sd_repo_configs_path, "v2-inference-v.yaml")
config_sd2_inpainting = os.path.join(
    sd_repo_configs_path, "v2-inpainting-inference.yaml"
)
config_sdxl = os.path.join(sd_xl_repo_configs_path, "sd_xl_base.yaml")
config_sdxl_refiner = os.path.join(sd_xl_repo_configs_path, "sd_xl_refiner.yaml")
config_depth_model = os.path.join(sd_repo_configs_path, "v2-midas-inference.yaml")
config_unclip = os.path.join(
    sd_repo_configs_path, "v2-1-stable-unclip-l-inference.yaml"
)
config_unopenclip = os.path.join(
    sd_repo_configs_path, "v2-1-stable-unclip-h-inference.yaml"
)
config_inpainting = os.path.join(sd_configs_path, "v1-inpainting-inference.yaml")
config_instruct_pix2pix = os.path.join(sd_configs_path, "instruct-pix2pix.yaml")
config_alt_diffusion = os.path.join(sd_configs_path, "alt-diffusion-inference.yaml")


def check_if_sdxl_from_state_dict(sd):
    if sd.get("conditioner.embedders.1.model.ln_final.weight", None) is not None:
        return True
    if sd.get("conditioner.embedders.0.model.ln_final.weight", None) is not None:
        return True
    return False


def load_safetensor_headers(path):
    left_curly_brackets_count = 0
    right_curly_brackets_count = 0
    with open(path, "rb") as f:
        f.seek(8)
        serialized_headers = bytearray()
        while (
            left_curly_brackets_count == 0
            or right_curly_brackets_count == 0
            or left_curly_brackets_count > right_curly_brackets_count
        ):
            c = f.read(1)
            if c == b"{":
                left_curly_brackets_count += 1
            elif c == b"}":
                right_curly_brackets_count += 1
            serialized_headers.extend(c)

        return json.loads(serialized_headers.decode(encoding="utf-8"))


def check_if_sdxl(path):
    safetensors_headers = load_safetensor_headers(path)
    return check_if_sdxl_from_state_dict(safetensors_headers)


if __name__ == "__main__":
    print(check_if_sdxl("models/Stable-diffusion/sdXL_v10RefinerVAEFix.safetensors"))
    print(check_if_sdxl("models/Stable-diffusion/xxmix9realistic_v40.safetensors"))
