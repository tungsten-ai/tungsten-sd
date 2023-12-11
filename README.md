# Tungsten Stable Diffusion Template
This repository is for creating your own Tungsten model with a stable diffusion checkpoint. 

Using this template, you can create a Stable Diffusion model in tungsten including followings:
- Basic functionalities in [automatic1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (e.g. prompt syntax)
- ControlNet - reference only, openpose & depth
- LoRA
- Textual inversion
- Default prompt and negative prompt

## For Windows
1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Download `tungsten-sd.zip` from [v0.1.1](https://github.com/tungsten-ai/tungsten-sd/releases/tag/v0.1.1) and extract its contents
3. Run `update.bat`
4. Put your SD files into following directories
    - 1 Checkpoint: `tungsten-sd/models/Stable-diffusion`
    - LoRAs (optional): `tungsten-sd/models/Lora`
    - embeddings (optional): `tungsten-sd/embeddings`
    - VAEs (optional): `tungsten-sd/models/VAE`
5. Run `build_and_push.bat` and enter the responses (e.g. username and password at [tungsten.run](https://tungsten.run)).

## For Linux

### Prerequisites

- Stable diffusion weights
- [Python 3.7+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/)

### Build your Stable Diffusion model
#### 0. Clone this repository
```
git clone --recursive https://github.com/tungsten-ai/tungsten-sd.git
cd tungsten-sd
```

#### 1. Install Tungstenkit

First, install [Tungstenkit](https://github.com/tungsten-ai/tungstenkit):

```bash
pip install tungstenkit
```

#### 2. Prepare weights
Put your Stable Diffusion model weights to ``models/Stable-diffusion``.

If you want to have your own LoRA and VAE, refer to [advanced configuration](#advanced-configuration).

#### 3. Build model

```bash
tungsten build . -n tungsten-stable-diffusion
```

### Push your model & run remotely

#### 1. Create a project on Tungsten

Go to [tungsten.run](https://tungsten.run/new) and create a project.

#### 2. Push the model to Tungsten

Log in to Tungsten:

```bash
tungsten login
```

Add tag of the model:
```bash
# Example: tungsten tag tungsten-stable-diffusion myproject:v1
tungsten tag tungsten-stable-diffusion <YOUR_PROJECT_NAME>:<YOUR_MODEL_VERSION>
```

Then, push the model to the project:
```bash
tungsten push <YOUR_PROJECT_NAME>
```

#### 3. Run the model on Tungsten

Visit [tungsten.run](https://tungsten.run) and go to the project page.

### Run your model locally

#### 0. Make your GPU visible to docker
Refer to this page: [https://tungsten-ai.github.io/docs/running_models/using_gpus/](https://tungsten-ai.github.io/docs/running_models/using_gpus/)

#### 1. Start demo
```bash
tungsten demo tungsten-stable-diffusion
```

### Advanced configuration
#### LoRA
1. Put your LoRA model weights to ``models/Lora``.
2. Modify ``StableDiffusion.get_loras`` function in ``tungsten_model.py`` to adjust the lora magnitude.

#### VAE
Put your VAE model weights to ``models/VAE``.

#### Embedding
1. Put your embedding files to ``embeddings``.
2. Customize prompt (see [Prompt customization](#prompt-customization))

#### Prompt customization
Modify following functions in ``tungsten_model.py``:
- ``StableDiffusion.get_trigger_words`` - Add trigger words at the start of the prompt.
- ``StableDiffusion.get_extra_prompt_chunks`` - Add extra prompt chunks at the end of the prompt.
- ``StableDiffusion.get_extra_negative_prompt_chunks`` - Add extra negative prompt chunks at the end of the negative prompt.
