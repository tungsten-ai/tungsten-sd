import importlib
import os
import sys
import warnings
from threading import Thread

os.environ["HF_HOME"] = "models/huggingface"
os.environ["TORCH_HOME"] = "models/torch"


warnings.filterwarnings("ignore")


def initialize(vae_file_path, *, is_sdxl):
    if is_sdxl:
        sys.argv.extend(["--xformers", "--no-half-vae"])
    else:
        sys.argv.extend(["--xformers"])

    # isort: off
    import torch
    import pytorch_lightning  # noqa: F401 # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
    from modules import paths, timer, import_hook, errors, devices  # noqa: F401

    from modules import shared, sd_samplers, upscaler, extensions
    import modules.codeformer_model as codeformer
    import modules.face_restoration
    import modules.gfpgan_model as gfpgan

    import modules.lowvram
    import modules.scripts
    import modules.sd_hijack
    import modules.sd_hijack_optimizations
    import modules.sd_models
    import modules.sd_vae
    import modules.sd_unet
    import modules.txt2img
    import modules.script_callbacks
    import modules.textual_inversion.textual_inversion
    from modules import extra_networks

    from modules import modelloader
    from modules.sd_vae import load_vae
    from modules.shared import cmd_opts

    def initialize_rest(*, reload_script_modules=False):
        """
        Called both from initialize() and when reloading the webui.
        """
        sd_samplers.set_samplers()
        extensions.list_extensions()

        if cmd_opts.ui_debug_mode:
            shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
            modules.scripts.load_scripts()
            return

        modules.sd_models.list_models()
        modules.scripts.load_scripts()

        if reload_script_modules:
            for module in [
                module
                for name, module in sys.modules.items()
                if name.startswith("modules.ui")
            ]:
                importlib.reload(module)

        modelloader.load_upscalers()

        modules.sd_vae.refresh_vae_list()
        modules.textual_inversion.textual_inversion.list_textual_inversion_templates()

        modules.script_callbacks.on_list_optimizers(
            modules.sd_hijack_optimizations.list_optimizers
        )
        modules.sd_hijack.list_optimizers()

        modules.sd_unet.list_unets()

        def load_model():
            """
            Accesses shared.sd_model property to load model.
            After it's available, if it has been loaded before this access by some extension,
            its optimization may be None because the list of optimizaers has neet been filled
            by that time, so we apply optimization again.
            """

            shared.sd_model  # noqa: B018

            if modules.sd_hijack.current_optimizer is None:
                modules.sd_hijack.apply_optimizations()

        th = Thread(target=load_model)
        th.start()

        shared.reload_hypernetworks()

        extra_networks.initialize()
        extra_networks.register_default_extra_networks()

        th.join()

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    initialize_rest()

    if vae_file_path:
        load_vae(vae_file_path)
