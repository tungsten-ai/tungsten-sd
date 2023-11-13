import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import click
import urllib3
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, Prompt
from tungsten_model_ast import TungstenModelAST


@click.command
def build_and_push():
    tungsten_model_ast = TungstenModelAST()

    console = Console(soft_wrap=True)

    # Get username and password
    console.print("Login to tungsten.run")
    username = Prompt.ask("Enter username or email", console=console)
    password = Prompt.ask("Enter password", password=True, console=console)
    _call_tungstenkit_cli(["login", "-u", username, "-p", password])
    console.print()

    # Get project name
    project_name = input("Enter project name: ")
    if not project_name:
        raise click.ClickException("Project name cannot be empty")
    _call_tungstenkit_cli(["project", "create", project_name, "--exists-ok"])
    console.print()

    # Get version
    version = input("Enter version (press ENTER to skip): ")
    console.print()

    # Select checkpoint
    ckpt_dir = Path("models/Stable-diffusion")
    checkpoint_paths = list(ckpt_dir.glob("*.safetensors"))
    if len(checkpoint_paths) == 0:
        raise click.ClickException(
            f"No checkpoint found. Please put a checkpoint in {ckpt_dir}."
        )
    if len(checkpoint_paths) > 1:
        raise click.ClickException(
            f"Too many checkpoints found. Plase put only one checkpoint in {ckpt_dir}"
        )
    checkpoint_filename = checkpoint_paths[0].name
    console.print(f"Checkpoint: {checkpoint_filename}")
    console.print()

    # Get LoRA info
    lora_dir = Path("models/Lora")
    lora_paths = list(lora_dir.glob("*.safetensors")) + list(lora_dir.glob("*.pt"))
    for lora_path in lora_paths:
        lora_filename = lora_path.name
        lora_filename_without_extension = _remove_extension_in_filename(lora_filename)
        console.print(f"Found LoRA: {lora_filename}")
        is_adjustable = Confirm.ask("Do you want to make a slide bar for this LoRA?")

        lora_strength, lora_expr = None, None
        if is_adjustable:
            verified_lora_filename_without_extension = re.sub(
                r"[^A-Za-z0-9_]+",
                "",
                lora_filename_without_extension.strip()
                .replace(" ", "_")
                .replace("-", "_"),
            )
            field_name = Prompt.ask(
                "Enter the name of the slide bar",
                default=verified_lora_filename_without_extension,
                show_default=True,
                console=console,
            )
            field_name = re.sub(r"[^A-Za-z0-9_]+", "", field_name).replace("-", "_")
            if field_name[0].isdigit():
                field_name = "_" + field_name
            ge = FloatPrompt.ask(
                "Enter the minimum value",
                default=0.0,
                show_default=True,
                console=console,
            )
            le = FloatPrompt.ask(
                "Enter the maximum value",
                default=1.0,
                show_default=True,
                console=console,
            )
            description = Prompt.ask(
                "Enter description for this LoRA (Press ENTER to skip)",
                default="",
                show_default=False,
                console=console,
            )
            tungsten_model_ast.add_optional_input_field(
                name=field_name,
                typename="float",
                default=0.0,
                description=description if description else field_name,
                le=le,
                ge=ge,
            )
            lora_expr = field_name
        else:
            lora_strength = FloatPrompt.ask(
                "Enter the strength of this LoRA", default=1.0, console=console
            )

        lora_trigger_words = Prompt.ask(
            "Enter trigger words (Press ENTER to skip)",
            default="",
            show_default=False,
            console=console,
        )

        tungsten_model_ast.add_lora(
            name=lora_filename_without_extension,
            magnitude=lora_strength,
            expr=lora_expr,
        )
        if lora_trigger_words:
            tungsten_model_ast.add_triger_word(lora_trigger_words)

        console.print()

    # Get embedding info
    embedding_dir = Path("embeddings")
    embedding_paths = list(embedding_dir.glob("*.safetensors")) + list(
        embedding_dir.glob("*.pt")
    )
    for embedding_path in embedding_paths:
        embedding_filename = embedding_path.name
        embedding_filename_without_extension = _remove_extension_in_filename(
            embedding_filename
        )
        console.print(
            f"Found embedding (textual inversion): {embedding_filename_without_extension}"
        )
        is_negative = Confirm.ask(
            "Is this a negative embedding?",
            console=console,
            default=True,
            show_default=True,
        )
        embedding_strength = FloatPrompt.ask(
            "Enter the strength of this embedding", default=1.0, console=console
        )

        if is_negative:
            tungsten_model_ast.add_extra_negative_prompt_chunk(
                value=embedding_filename_without_extension, magnitude=embedding_strength
            )
        else:
            tungsten_model_ast.add_extra_prompt_chunk(
                value=embedding_filename_without_extension, magnitude=embedding_strength
            )

        console.print()

    # Get extra positive/negative prompt
    extra_prompt = Prompt.ask(
        "Enter extra prompt. This is appended at the end of the user prompt. No need to add embedding keywords. (Press ENTER to skip)",
        default="",
        show_default=False,
        console=console,
    )
    console.print()
    extra_negative_prompt = Prompt.ask(
        "Enter extra negative prompt. This is appended at the end of the user negative prompt. No need to add embedding keywords. (Press ENTER to skip)",
        default="",
        show_default=False,
        console=console,
    )
    console.print()
    if extra_prompt:
        tungsten_model_ast.add_extra_prompt_chunk(extra_prompt)
    if extra_negative_prompt:
        tungsten_model_ast.add_extra_negative_prompt_chunk(extra_negative_prompt)

    console.print()

    tungsten_model_fd, tungsten_model_path_str = tempfile.mkstemp(
        prefix="tungsten_model_", suffix=".py", dir=".", text=True
    )
    try:
        os.close(tungsten_model_fd)
        # Create tungsten_model_xxxxxxx.py
        tungsten_model_path = Path(tungsten_model_path_str)
        tungsten_model_path.write_text(tungsten_model_ast.unparse())

        # Build
        tungsten_model_module_ref = tungsten_model_path.name.replace(".py", "")
        full_model_name = f"{project_name}:{version}" if version else project_name
        console.print("Start building the model")
        _call_tungstenkit_cli(
            [
                "build",
                os.path.dirname(__file__),
                "-m",
                tungsten_model_module_ref,
                "-n",
                full_model_name,
            ]
        )
        try:
            # Push
            console.print("Upload the model to tungsten.run")
            _call_tungstenkit_cli(["push", project_name])

        finally:
            # Remove docker image
            console.print("Removing model files stored in this machine...")
            _call_tungstenkit_cli(["clear", project_name])

    finally:
        # Remove tungsten model module
        if os.path.exists(tungsten_model_path_str):
            os.remove(tungsten_model_path_str)


def _remove_extension_in_filename(filename: str):
    parts = filename.split(".")
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return filename


def _call_tungstenkit_cli(cmd: List):
    try:
        subprocess.run(
            [sys.executable, "-m", "tungstenkit._internal.cli.main"] + cmd,
            check=True,
            cwd=os.path.dirname(__file__),
        )
    except subprocess.SubprocessError:
        sys.exit(1)


if __name__ == "__main__":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    build_and_push()
