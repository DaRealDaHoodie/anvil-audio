"""
Launch the Stable Audio Tools Gradio web interface.

Examples::

    # No arguments — loads the first model from the registry
    python run_gradio.py

    # Registry name shorthand
    python run_gradio.py --model stable-audio-open-small

    # Pretrained model from HuggingFace Hub
    python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0

    # Local config + checkpoint
    python run_gradio.py --model-config config.json --ckpt-path model.ckpt

    # Route outputs to a named project folder
    python run_gradio.py --model stable-audio-open-1.0 --project sfx-pack-v1

    # Apple Silicon
    python run_gradio.py --model stable-audio-open-1.0 --device mps

    # Create a public share URL (off by default)
    python run_gradio.py --model stable-audio-open-1.0 --share
"""

from __future__ import annotations

import argparse
import sys

import torch

from anvil_audio.core.registry import registry
from anvil_audio.interface.gradio import create_ui
from anvil_audio.utils.torch_common import get_best_device


def resolve_model(args: argparse.Namespace) -> str | None:
    """Return the pretrained_name to load, or None if using local config/ckpt.

    Resolution order:
    1. --model (registry name lookup)
    2. --pretrained-name (raw HF repo ID, passed through as-is)
    3. --model-config + --ckpt-path (local files, returns None)
    4. No args → first registry entry, or error if registry is empty
    """
    if args.model:
        entry = registry.get_model(args.model)
        if entry is None:
            available = ", ".join(e.name for e in registry.list_models()) or "(none)"
            print(
                f"error: --model '{args.model}' not found in registry.\n"
                f"Available models: {available}\n"
                f"Add your own in ~/.anvil-audio/registry.yaml",
                file=sys.stderr,
            )
            sys.exit(1)
        if entry.pretrained_name:
            return entry.pretrained_name
        # local-config entry — handled via model_config_path/ckpt_path
        args.model_config = args.model_config or entry.model_config_path
        args.ckpt_path = args.ckpt_path or entry.ckpt_path
        return None

    if args.pretrained_name:
        return args.pretrained_name

    if args.model_config and args.ckpt_path:
        return None

    # No model specified — fall back to first registry entry
    models = registry.list_models()
    if not models:
        print(
            "error: no model specified and the registry is empty.\n\n"
            "Pass a model with one of:\n"
            "  --model <registry-name>          (e.g. --model stable-audio-open-1.0)\n"
            "  --pretrained-name <hf-repo-id>   (e.g. --pretrained-name stabilityai/stable-audio-open-1.0)\n"
            "  --model-config <path> --ckpt-path <path>\n\n"
            "Or add a model to ~/.anvil-audio/registry.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    default = models[0]
    print(f"->->-> No model specified — using registry default: {default.name}")
    if default.pretrained_name:
        return default.pretrained_name
    args.model_config = args.model_config or default.model_config_path
    args.ckpt_path = args.ckpt_path or default.ckpt_path
    return None


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(42)

    device = torch.device(args.device) if args.device else get_best_device()
    print(f"->->-> Using device: {device}")

    pretrained_name = resolve_model(args)

    interface = create_ui(
        model_config_path=args.model_config,
        ckpt_path=args.ckpt_path,
        pretrained_name=pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half,
        tmp_dir=args.tmp_dir,
        device=device,
        project=args.project,
    )
    interface.queue()
    from pathlib import Path
    output_root = str(Path.home() / "anvil-audio-outputs")
    interface.launch(
        share=args.share,
        auth=(args.username, args.password) if args.username is not None else None,
        allowed_paths=[output_root],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Anvil Audio Gradio interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, help="Registry model name (e.g. stable-audio-open-small). Takes precedence over --pretrained-name.", required=False)
    parser.add_argument("--pretrained-name", type=str, help="HuggingFace Hub model repo ID", required=False)
    parser.add_argument("--model-config", type=str, help="Path to model config JSON", required=False)
    parser.add_argument("--ckpt-path", type=str, help="Path to model checkpoint", required=False)
    parser.add_argument("--pretransform-ckpt-path", type=str, help="Optional path to pretransform checkpoint", required=False)
    parser.add_argument("--username", type=str, help="Gradio auth username", required=False)
    parser.add_argument("--password", type=str, help="Gradio auth password", required=False)
    parser.add_argument("--model-half", action="store_true", help="Use float16 (half precision) inference")
    parser.add_argument("--tmp-dir", type=str, default="", help="Legacy parameter (unused; output manager handles paths)")
    parser.add_argument("--device", type=str, default="", help="Device override: cuda, mps, cpu (auto-detects if not set)")
    parser.add_argument("--project", type=str, default="", help="Project name — outputs go to ~/anvil-audio-outputs/{project}/")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL")
    main(parser.parse_args())
