"""
Launch the Stable Audio Tools Gradio web interface.

Examples::

    # Pretrained model from HuggingFace Hub
    python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0

    # Local config + checkpoint
    python run_gradio.py --model-config config.json --ckpt-path model.ckpt

    # Route outputs to a named project folder
    python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0 --project sfx-pack-v1

    # Apple Silicon
    python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0 --device mps

    # Create a public share URL (off by default)
    python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0 --share
"""

from __future__ import annotations

import argparse

import torch

from anvil_audio.interface.gradio import create_ui
from anvil_audio.utils.torch_common import get_best_device


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(42)

    device = torch.device(args.device) if args.device else get_best_device()
    print(f"->->-> Using device: {device}")

    interface = create_ui(
        model_config_path=args.model_config,
        ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
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
    parser = argparse.ArgumentParser(description="Run the Stable Audio Tools Gradio interface")
    parser.add_argument("--pretrained-name", type=str, help="HuggingFace Hub model repo ID", required=False)
    parser.add_argument("--model-config", type=str, help="Path to model config JSON (ignored if --pretrained-name is set)", required=False)
    parser.add_argument("--ckpt-path", type=str, help="Path to model checkpoint (ignored if --pretrained-name is set)", required=False)
    parser.add_argument("--pretransform-ckpt-path", type=str, help="Optional path to pretransform checkpoint", required=False)
    parser.add_argument("--username", type=str, help="Gradio auth username", required=False)
    parser.add_argument("--password", type=str, help="Gradio auth password", required=False)
    parser.add_argument("--model-half", action="store_true", help="Use float16 (half precision) inference")
    parser.add_argument("--tmp-dir", type=str, default="", help="Legacy parameter (unused; output manager handles paths)")
    parser.add_argument("--device", type=str, default="", help="Device override: cuda, mps, cpu (auto-detects if not set)")
    parser.add_argument("--project", type=str, default="", help="Project name — outputs go to ~/anvil-audio-outputs/{project}/")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL")
    main(parser.parse_args())
