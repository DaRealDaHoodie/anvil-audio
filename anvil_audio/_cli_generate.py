"""
Generate audio samples using a registered or locally-configured model.

Usage — registry (preferred)
-----------------------------
    anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"
    anvil generate --model sfx-v1 --cond-yaml-path batch.yaml --output-dir ./out
    anvil generate --list-models

Usage — legacy config path (unchanged behaviour)
-------------------------------------------------
    anvil generate --model-config path/to/config.json --ckpt-path path/to/ckpt.pt \
        --prompt "wooden door creak" --output-dir ./out

Multi-GPU is supported via Accelerate for both paths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import yaml
from pathlib import Path
from typing import Any

import torch
import torchaudio
from accelerate import Accelerator

from anvil_audio.core import load_pipeline, registry
from anvil_audio.core.pipeline import DiffusionPipeline
from anvil_audio.utils.torch_common import count_parameters, get_rank, get_world_size, get_best_device
from anvil_audio.utils.audio_utils import float_to_int16_audio

SUPPORTED_FORMATS = ("wav", "flac", "mp3")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate audio samples from a generative audio model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Model selection (mutually exclusive groups) ----
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        metavar="NAME",
        help="Registry model name (e.g. 'stable-audio-open-1.0'). "
             "Use --list-models to see available names.",
    )
    model_group.add_argument(
        "--list-models",
        action="store_true",
        help="Print all registered models and exit.",
    )

    # ---- Legacy config path ----
    p.add_argument(
        "--model-config",
        type=str,
        metavar="PATH",
        help="Path to a JSON model config (legacy path; ignored if --model is set).",
    )
    p.add_argument(
        "--ckpt-path",
        type=str,
        metavar="PATH",
        help="Path to model checkpoint (legacy path; required with --model-config).",
    )
    p.add_argument(
        "--pretransform-ckpt-path",
        type=str,
        metavar="PATH",
        help="Optional separate checkpoint for the pretransform / VAE stage.",
    )

    # ---- Prompt / condition input ----
    prompt_group = p.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        metavar="TEXT",
        help="Single text prompt.  Use instead of --cond-yaml-path for quick runs.",
    )
    prompt_group.add_argument(
        "--cond-yaml-path",
        type=str,
        metavar="PATH",
        help="Path to a YAML file containing batch conditions.",
    )

    p.add_argument(
        "--seconds-start",
        type=float,
        default=0.0,
        help="Start time in seconds (used with --prompt). Default: 0.0",
    )
    p.add_argument(
        "--seconds-total",
        type=float,
        default=30.0,
        help="Total duration in seconds (used with --prompt). Default: 30.0",
    )

    # ---- Output ----
    p.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for saved audio files. Default: ./output",
    )
    p.add_argument(
        "--format",
        type=str,
        default="wav",
        choices=SUPPORTED_FORMATS,
        help="Output audio format. Default: wav",
    )
    p.add_argument(
        "--clip-length",
        action="store_true",
        help="Clip generated audio to 'seconds_total' from each condition.",
    )

    # ---- Sampling ----
    p.add_argument("--sample-steps", type=int, default=None,
                   help="Diffusion steps. Overrides registry/config default.")
    p.add_argument("--cfg-scale", type=float, default=None,
                   help="Classifier-free guidance scale. Overrides default.")
    p.add_argument("--sampler-type", type=str, default=None,
                   help="Sampler type. Overrides default.")
    p.add_argument("--sigma-min", type=float, default=None,
                   help="Sigma min. Overrides default.")
    p.add_argument("--sigma-max", type=float, default=None,
                   help="Sigma max. Overrides default.")
    p.add_argument("--n-sample-per-cond", type=int, default=1,
                   help="Number of samples per condition entry. Default: 1")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Max items per inference batch per GPU. Default: 10")
    p.add_argument("--seed", type=int, default=-1,
                   help="RNG seed. -1 uses a random seed. Default: -1")

    # ---- Device ----
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Device (cuda, mps, cpu). Auto-detects best if not set.",
    )

    return p


# ---------------------------------------------------------------------------
# Condition loading helpers
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/", depth: int = 0) -> dict:
    items: dict = {}
    for k, v in d.items():
        if depth == 0:
            assert isinstance(v, dict) and all(isinstance(v_, dict) for v_ in v.values())
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(list(v.values())[0], dict):
            items.update(_flatten_dict(v, new_key, sep=sep, depth=depth + 1))
        else:
            assert all(not isinstance(v_, dict) for v_ in v.values())
            items[new_key] = dict(v)
    return items


def _load_conditions(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Return an ordered dict of {path_key: condition_dict}."""
    if args.prompt:
        return {
            "prompt/item": {
                "prompt": args.prompt,
                "seconds_start": args.seconds_start,
                "seconds_total": args.seconds_total,
            }
        }
    with open(args.cond_yaml_path) as fh:
        raw = yaml.safe_load(fh)
    conds = _flatten_dict(raw)
    for cond in conds.values():
        cond.setdefault("seconds_start", 0.0)
        cond.setdefault("seconds_total", args.seconds_total)
    return conds


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------

def _save_audio(audio: torch.Tensor, path: str, sample_rate: int, fmt: str) -> None:
    from anvil_audio.utils.audio_utils import save_audio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_audio(path, audio, sample_rate, fmt=fmt)


# ---------------------------------------------------------------------------
# Pipeline loader (handles both registry and legacy paths)
# ---------------------------------------------------------------------------

def _load_pipeline(args: argparse.Namespace, device: torch.device) -> DiffusionPipeline:
    if args.model:
        return load_pipeline(args.model, device=str(device))

    # Legacy path
    if not args.model_config:
        raise SystemExit(
            "error: one of --model, --model-config, or --list-models is required."
        )
    if not args.ckpt_path:
        raise SystemExit("error: --ckpt-path is required when using --model-config.")

    from anvil_audio.models.factory import create_pipeline_from_config
    with open(args.model_config) as fh:
        model_config = json.load(fh)

    pipeline = create_pipeline_from_config(
        model_config,
        ckpt_path=args.ckpt_path,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
    )
    pipeline.to(device)
    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --list-models: print and exit (no model loading needed)
    if args.list_models:
        entries = registry.list_models()
        if not entries:
            print("No models registered.")
        else:
            print(f"{'NAME':<35}  {'SOURCE'}")
            print("-" * 70)
            for e in entries:
                source = e.pretrained_name or e.model_config_path or "(local)"
                print(f"{e.name:<35}  {source}")
        sys.exit(0)

    # Validate prompt / condition input
    if not args.prompt and not args.cond_yaml_path:
        parser.error("one of --prompt or --cond-yaml-path is required for generation.")

    # Device
    device: torch.device
    if args.device:
        device = torch.device(args.device)
    else:
        accel = Accelerator()
        device = accel.device if accel.device.type != "cpu" else get_best_device()

    rank = get_rank()
    world_size = get_world_size()

    # Load pipeline
    pipeline = _load_pipeline(args, device)
    pipeline.eval()

    sample_rate = pipeline.sample_rate
    sample_size = pipeline.sample_size

    # Build per-call generation kwargs from CLI flags (only set if explicitly passed)
    gen_kwargs: dict[str, Any] = {}
    if args.cfg_scale is not None:
        gen_kwargs["cfg_scale"] = args.cfg_scale
    if args.sampler_type is not None:
        gen_kwargs["sampler_type"] = args.sampler_type
    if args.sigma_min is not None:
        gen_kwargs["sigma_min"] = args.sigma_min
    if args.sigma_max is not None:
        gen_kwargs["sigma_max"] = args.sigma_max

    steps = args.sample_steps  # None means use pipeline default

    # Effective batch window (CFG doubles the batch internally)
    cfg = gen_kwargs.get("cfg_scale", pipeline.default_params.get("cfg_scale", 7.0))
    batch_window = max(args.batch_size // 2, 1) if cfg != 1.0 else args.batch_size

    # Load conditions and expand by n_sample_per_cond
    conds = _load_conditions(args)
    path_full: list[str] = []
    conds_full: list[dict[str, Any]] = []
    for p, cond in conds.items():
        for idx in range(args.n_sample_per_cond):
            path_full.append(f"{p}_item-{idx + 1}")
            conds_full.append(cond)

    # Print info (main process only)
    if rank == 0:
        model = pipeline.model
        params_model = count_parameters(model.model)
        params_cond = count_parameters(model.conditioner)
        effective_params = {**pipeline.default_params, **gen_kwargs}

        print("=== Model Info ===")
        print(f"\tDevice:\t\t{device}")
        print(f"\tSample rate:\t{sample_rate}")
        print(f"\tSample size:\t{sample_size} ({sample_size / sample_rate:.3f} sec)")
        print("=== Parameters (M) ===")
        print(f"\tDiffusion:\t{params_model / 1e6:.3f}")
        print(f"\tConditioner:\t{params_cond / 1e6:.3f}")
        print("=== Generation ===")
        print(f"\tSteps:\t\t{steps or effective_params.get('steps', 100)}")
        print(f"\tCFG scale:\t{effective_params.get('cfg_scale', 7.0)}")
        print(f"\tSampler:\t{effective_params.get('sampler_type', 'dpmpp-3m-sde')}")
        print(f"\tSeed:\t\t{args.seed}")
        print(f"\tFormat:\t\t{args.format}")
        print(f"\tTotal items:\t{len(path_full)} ({len(conds)} prompts × {args.n_sample_per_cond})")
        print()

    # Distribute across ranks
    path_rank = path_full[rank::world_size]
    conds_rank = conds_full[rank::world_size]

    # Generation loop
    n_iter = math.ceil(len(conds_rank) / batch_window)
    for i in range(n_iter):
        path_i = path_rank[i * batch_window: (i + 1) * batch_window]
        conds_i = conds_rank[i * batch_window: (i + 1) * batch_window]

        samples = pipeline.generate(
            conditioning=conds_i,
            steps=steps,
            seed=args.seed,
            disable_tqdm=(rank != 0),
            **gen_kwargs,
        )

        for j in range(samples.shape[0]):
            audio = float_to_int16_audio(samples[j])
            if args.clip_length:
                L = int(conds_i[j]["seconds_total"] * sample_rate)
                audio = audio[:, :L]
            save_path = f"{args.output_dir}/{path_i[j]}.{args.format}"
            _save_audio(audio, save_path, sample_rate, args.format)

    print(f"->->-> Rank-{rank}: Finished.")


if __name__ == "__main__":
    main()
