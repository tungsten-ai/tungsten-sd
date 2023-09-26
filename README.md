# Tungsten Stable Diffusion Template
This repository is for creating your own Tungsten model with a custom stable diffusion checkpoint. 

Using this template, you can create a Tungsten model including followings:
- Basic functionalities in [automatic1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (e.g. prompt syntax & Real-ESRGAN upscaler)
- ControlNet - reference only, openpose & depth (currently available only for non-XL models)
- LoRA

## Prerequisites

- Stable diffusion weights
- [Python 3.7+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/)

## Create Stable Diffusion model in Tungsten
### Step 0: Install Tungstenkit

First, install [Tungstenkit](https://github.com/tungsten-ai/tungstenkit):

```bash
pip install tungstenkit
```

### Step 1. Prepare weights
Put your Stable Diffusion model weights to ``models/Stable-diffusion``.

If you want to have your own LoRA and VAE, refer to [advanced configuration](#advanced-configuration).

### Step 2. Build model

```bash
tungsten build . -n tungsten-stable-diffusion
```

### Step 3: Create a project on Tungsten

Go to [tungsten.run](https://tungsten.run/new) and create a project.

### Step 4: Push the model to Tungsten

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

### Step 5: Run the model on Tungsten

Visit [tungsten.run](https://tungsten.run) and go to the project page.


## Advanced Configuration
### LoRA
1. Put your LoRA model weights to ``models/Lora``.
2. Modify ``StableDiffusion.get_loras`` function in ``tungsten_model.py`` to adjust lora magnitude

### VAE
Put your VAE model weights to ``models/VAE``
