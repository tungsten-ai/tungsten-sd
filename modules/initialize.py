import os
import sys
import warnings
from threading import Thread

os.environ["HF_HOME"] = "models/huggingface"
os.environ["TORCH_HOME"] = "models/torch"
os.environ["SAFETENSORS_FAST_GPU"] = "1"


warnings.filterwarnings("ignore")


def initialize(*, is_sdxl, default_sampler):
    if is_sdxl:
        sys.argv.extend(
            [
                "--xformers",
                "--no-half-vae",
                "--controlnet-dir=models/ControlNet",
                "--controlnet-annotator-models-path=models/ControlNetAnnotators",
                "--no-hashing",
                "--ad-no-huggingface",
            ]
        )
    else:
        sys.argv.extend(
            [
                "--xformers",
                "--controlnet-dir=models/ControlNet",
                "--controlnet-annotator-models-path=models/ControlNetAnnotators",
                "--no-hashing",
                "--ad-no-huggingface",
            ]
        )

    # isort: off
    import multiprocessing as mp
    import torch

    cpu_count = mp.cpu_count()

    torch.set_num_threads(cpu_count - 1)
    torch.set_num_interop_threads(cpu_count - 1)

    import pytorch_lightning  # noqa: F401 # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
    from modules import paths, timer, import_hook, errors, shared, rng  # noqa: F401

    from modules import sd_samplers, upscaler, extensions

    import modules.codeformer_model as codeformer
    import modules.face_restoration
    import modules.gfpgan_model as gfpgan

    import modules.devices

    import modules.lowvram
    import modules.scripts
    import modules.sd_models
    import modules.sd_vae
    import modules.sd_unet
    import modules.txt2img
    import modules.script_callbacks
    from modules import extra_networks
    import modules.processing

    import modules.sd_hijack
    import modules.sd_hijack_optimizations
    from modules.textual_inversion import textual_inversion
    from modules import modelloader
    from modules.sd_vae import load_vae
    from modules import sd_unet
    from modules.shared import cmd_opts
    from modules.processing import decode_first_stage

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """
        print("Started loading weights")

        shared.sd_model  # noqa: B018

        if modules.sd_hijack.current_optimizer is None:
            modules.sd_hijack.apply_optimizations()

    t = timer.Timer()

    modelloader.cleanup_models()
    t.record("cleanup models")

    modules.sd_models.setup_model()
    t.record("setup model")

    sd_samplers.set_samplers()
    t.record("set samplers")

    extensions.list_extensions()
    t.record("list extensions")

    modules.scripts.load_scripts()
    t.record("load scripts")

    modules.sd_models.list_models()
    t.record("list models")

    modelloader.load_upscalers()
    t.record("list upsamplers")

    modules.sd_vae.refresh_vae_list()
    t.record("refresh vae list")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    t.record("load textual embeddings")

    th_initialize_scripts = Thread(
        target=modules.scripts.scripts_txt2img.initialize_scripts,
        kwargs={"is_img2img": False},
    )
    th_initialize_scripts.start()

    modules.script_callbacks.on_list_optimizers(
        modules.sd_hijack_optimizations.list_optimizers
    )
    modules.sd_hijack.list_optimizers()
    t.record("list optimizers")

    modules.sd_unet.list_unets()
    t.record("list unets")

    th_loading_weights = Thread(target=load_model)
    th_loading_weights.start()

    shared.reload_hypernetworks()
    t.record("reload hypernetworks")

    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    t.record("initialize extra networks")

    th_loading_weights.join()
    th_initialize_scripts.join()
    t.record("wait for loading model")

    modules.script_callbacks.before_ui_callback()
    modules.script_callbacks.ui_settings_callback()
    t.record("init callbacks")

    print(f"Setup done in {t.summary()}")


def load_vae_weights(vae_file_path):
    from modules import shared
    from modules.sd_vae import load_vae

    if vae_file_path:
        load_vae(shared.sd_model, vae_file_path)


def initialize_vae():
    from modules import shared
    from modules.sd_vae import clear_loaded_vae, restore_base_vae

    clear_loaded_vae()
    restore_base_vae(shared.sd_model)
