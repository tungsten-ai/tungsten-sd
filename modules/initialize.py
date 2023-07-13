import os
import warnings
from contextlib import redirect_stdout

os.environ["HF_HOME"] = "models/huggingface"
os.environ["TORCH_HOME"] = "models/torch"
import modules.paths

warnings.filterwarnings("ignore")


def initialize():
    with open(os.devnull, "w") as f:
        with redirect_stdout(f):
            import modules.hashes
            import modules.script_callbacks
            import modules.scripts
            import modules.sd_hijack
            import modules.sd_hijack_optimizations
            import modules.sd_models
            import modules.sd_unet
            import modules.sd_vae
            from modules import modelloader, sd_samplers

            modelloader.cleanup_models()
            modules.sd_models.setup_model()
            sd_samplers.set_samplers()
            modules.sd_models.list_models()
            modules.scripts.load_scripts()
            modules.sd_vae.refresh_vae_list()
            modules.script_callbacks.on_list_optimizers(
                modules.sd_hijack_optimizations.list_optimizers
            )
            modules.sd_hijack.list_optimizers()
            modules.sd_unet.list_unets()
            modules.script_callbacks.before_ui_callback()

    modules.sd_models.load_model()
