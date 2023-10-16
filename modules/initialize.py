import os
import sys
import warnings
from threading import Thread

os.environ["HF_HOME"] = "models/huggingface"
os.environ["TORCH_HOME"] = "models/torch"
os.environ["SAFETENSORS_FAST_GPU"] = "1"


warnings.filterwarnings("ignore")


def initialize(vae_file_path, *, is_sdxl, default_sampler):
    if is_sdxl:
        sys.argv.extend(["--xformers", "--no-half-vae", "--no-hashing"])
    else:
        sys.argv.extend(
            [
                "--xformers",
                "--controlnet-dir=models/ControlNet",
                "--controlnet-annotator-models-path=models/ControlNetAnnotators",
                "--no-hashing",
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

    if vae_file_path:
        load_vae(shared.sd_model, vae_file_path)
        print(f"Using custom VAE: {vae_file_path}")
        t.record("load vae")

    modules.script_callbacks.before_ui_callback()
    t.record("init callbacks")

    # occupy_mem(0)
    # t.record("preoccupy GPU memory")

    # print("Compiling computational graph of decoder...")
    # shape = (1, 4, 160, 96) if is_sdxl else (1, 4, 1, 1)
    # decode_first_stage(
    #     shared.sd_model,
    #     torch.rand(*shape).to(modules.devices.device).to(modules.devices.dtype_vae),
    # )
    # print("done.")
    # t.record("compile decoder graph")

    # print("Compiling computational graph of unet")
    # p = modules.processing.StableDiffusionProcessingTxt2Img(
    #     sd_model=shared.sd_model,
    #     prompt="a",
    #     negative_prompt="a",
    #     seed=0,
    #     sampler_name=default_sampler,
    #     batch_size=1,
    #     steps=1,
    #     cfg_scale=7,
    #     width=16,
    #     height=16,
    #     override_settings={},
    # )
    # # modules.processing.process_images(p)
    # # p.close()
    # with modules.devices.without_autocast() if modules.devices.unet_needs_upcast else modules.devices.autocast():
    #     n = 0
    #     p.all_seeds = p.all_subseeds = [0]
    #     p.setup_prompts()
    #     p.prompts = p.all_prompts[n * p.batch_size : (n + 1) * p.batch_size]
    #     p.negative_prompts = p.all_negative_prompts[
    #         n * p.batch_size : (n + 1) * p.batch_size
    #     ]
    #     p.seeds = p.all_seeds[n * p.batch_size : (n + 1) * p.batch_size]
    #     p.subseeds = p.all_subseeds[n * p.batch_size : (n + 1) * p.batch_size]
    #     opt_C = 4
    #     opt_f = 8
    #     p.rng = rng.ImageRNG(
    #         (opt_C, p.height // opt_f, p.width // opt_f),
    #         p.seeds,
    #         subseeds=p.subseeds,
    #         subseed_strength=p.subseed_strength,
    #         seed_resize_from_h=p.seed_resize_from_h,
    #         seed_resize_from_w=p.seed_resize_from_w,
    #     )
    #     with torch.no_grad(), p.sd_model.ema_scope():
    #         with modules.devices.autocast():
    #             p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
    #             sd_unet.apply_unet()
    #             p.parse_extra_network_prompts()
    #             p.setup_conds()
    #             p.sample(
    #                 conditioning=p.c,
    #                 unconditional_conditioning=p.uc,
    #                 seeds=p.seeds,
    #                 subseeds=p.subseeds,
    #                 subseed_strength=p.subseed_strength,
    #                 prompts=p.prompts,
    #             )
    #             p.close()

    # print("done.")
    # t.record("compile unet graph")

    print(f"Setup done in {t.summary()}")
    # print(torch.backends.cudnn.benchmark)
