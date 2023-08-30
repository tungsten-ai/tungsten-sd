# Tungsten Stable Diffusion
This repository is for creating your own Tungsten model with a custom stable diffusion checkpoint.

## Prerequisites

- Stable diffusion weights
- [Python 3.7+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/)

## Step 0: Install Tungstenkit

First, install [Tungstenkit](https://github.com/tungsten-ai/tungstenkit):

```bash
pip install tungstenkit
```

## Step 1. Prepare weights
Put your Stable Diffusion model weights to ``models/Stable-diffusion``. Then, download weights of extra models:
```
./download_extra_weights.sh
```

## Step 2. Build model

```bash
tungsten build . -n tungsten-stable-diffusion
```

## Step 3: Create a project on Tungsten

Go to [tungsten.run](https://tungsten.run/new) and create a project.

## Step 4: Push the model to Tungsten

Log in to Tungsten:

```bash
tungsten login
```

Add tag of the model:
```bash
tungsten tag tungsten-stable-diffusion <project_name>
```

Push the model to the project:
```bash
tungsten push <project_name>
```


## Step 5: Run the model on Tungsten

Visit [tungsten.run](https://tungsten.run) and go to the project page.
