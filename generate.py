"""
Copyright (C) 2024 Yukara Ikemiya

Generate audio samples using a pretrained generative model.
This script supports multi-GPU processing via Accelerate, and also runs
on Apple Silicon (MPS) or CPU for single-process use.
"""

import argparse
import math
import yaml
from pathlib import Path

import torch
import torchaudio
from accelerate import Accelerator

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.utils.torch_common import count_parameters, get_rank, get_world_size, get_best_device
from stable_audio_tools.utils.audio_utils import float_to_int16_audio

SUPPORTED_FORMATS = ["wav", "flac", "mp3"]


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--output-dir', type=str, required=True, help="Directory for saving generated audio samples.")
    args.add_argument('--model-name', type=str, default="stabilityai/stable-audio-open-1.0", help="Pretrained model name.")
    args.add_argument('--sampler-type', type=str, default="dpmpp-3m-sde", help="Diffusion sampler type.")
    args.add_argument('--sample-steps', type=int, default=100, help="Number of diffusion steps.")
    args.add_argument('--cfg-scale', type=float, default=7.0, help="Classifier-free guidance scale.")
    args.add_argument('--n-sample-per-cond', type=int, default=1, help="Number of samples per condition.")
    args.add_argument('--batch-size', type=int, default=10, help="Batch size per GPU.")
    args.add_argument('--clip-length', action='store_true', help="Clip output to 'seconds_total'.")
    args.add_argument('--seed', type=int, default=-1, help="Random seed (-1 for random).")
    args.add_argument('--device', type=str, default='', help="Device to use (cuda, mps, cpu). Auto-detects if not set.")
    args.add_argument('--format', type=str, default='wav', choices=SUPPORTED_FORMATS, help="Output audio format.")
    # Inline prompt mode (alternative to --cond-yaml-path)
    args.add_argument('--cond-yaml-path', type=str, default='', help="YAML file of sample conditions.")
    args.add_argument('--prompt', type=str, default='', help="Single text prompt (no YAML needed).")
    args.add_argument('--seconds-start', type=float, default=0.0, help="Start time in seconds (used with --prompt).")
    args.add_argument('--seconds-total', type=float, default=30.0, help="Total duration in seconds (used with --prompt).")
    args = args.parse_args()
    return args


def flatten_dict(d, parent_key='', separator='/', depth=0):
    items = {}
    for k, v in d.items():
        if depth == 0:
            assert isinstance(v, dict) and all([isinstance(v_, dict) for v_ in v.values()])
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(list(v.values())[0], dict):
            items.update(flatten_dict(v, new_key, separator=separator, depth=depth + 1))
        else:
            assert all([not isinstance(v_, dict) for v_ in v.values()])
            items[new_key] = {k_: v_ for k_, v_ in v.items()}

    return items


def parse_cond_yaml(yaml_path):
    with open(yaml_path, 'r') as yml:
        conds: dict = yaml.safe_load(yml)

    conds: dict = flatten_dict(conds)
    return conds


def save_audio(audio: torch.Tensor, path: str, sample_rate: int, fmt: str):
    """Save audio tensor to file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "wav":
        torchaudio.save(path, audio, sample_rate)
    elif fmt == "flac":
        torchaudio.save(path, audio, sample_rate, format="flac")
    elif fmt == "mp3":
        torchaudio.save(path, audio, sample_rate, format="mp3")


def main():
    args = get_args()

    # Validate inputs
    if not args.cond_yaml_path and not args.prompt:
        raise ValueError("Must provide either --cond-yaml-path or --prompt")
    if args.cond_yaml_path and args.prompt:
        raise ValueError("Provide either --cond-yaml-path or --prompt, not both")

    # Config
    output_dir: str = args.output_dir
    model_name: str = args.model_name
    sampler_type: str = args.sampler_type
    sample_steps: int = args.sample_steps
    cfg_scale: float = args.cfg_scale
    n_sample_per_cond: int = args.n_sample_per_cond
    batch_size: int = args.batch_size
    clip_length: bool = args.clip_length
    seed: int = args.seed
    fmt: str = args.format

    batch_sample: int = max(batch_size // 2, 1) if cfg_scale != 1.0 else batch_size

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        # Accelerate handles multi-GPU; fall back to best device for single-process
        device = None  # resolved below after Accelerator init

    # Multi-GPU setup
    accel = Accelerator()
    rank = get_rank()
    world_size = get_world_size()

    if device is None:
        device = accel.device if accel.device.type != "cpu" else get_best_device()

    # Load model
    model, model_config = get_pretrained_model(model_name)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model: ConditionedDiffusionModelWrapper = model.to(device)

    # Build condition list
    if args.prompt:
        conds = {"prompt/item": {"prompt": args.prompt, "seconds_start": args.seconds_start, "seconds_total": args.seconds_total}}
    else:
        conds = parse_cond_yaml(args.cond_yaml_path)

    path_full = []
    conds_full = []
    for p, cond in conds.items():
        for idx in range(n_sample_per_cond):
            path_full.append(f"{p}_item-{idx + 1}")
            conds_full.append(cond)

    # Print info on main process
    if accel.is_main_process:
        model.train()
        params_model = count_parameters(model.model)
        params_cond = count_parameters(model.conditioner)

        print("=== Model Info ===")
        print(f"\tDevice:\t\t{device}")
        print(f"\tSample rate:\t{sample_rate}")
        print(f"\tOut channels:\t{model.pretransform.io_channels}")
        print(f"\tSample size:\t{sample_size} ({sample_size / sample_rate:.3f} [sec])")
        print("=== Model parameters ===")
        print(f'\tDiffusion :\t\t{params_model / (10**6):.3f} [million]')
        print(f'\tConditioner :\t{params_cond / (10**6):.3f} [million]')
        print("=== Sampling parameters ===")
        print(f"\tSampler type:\t{sampler_type}")
        print(f"\tSample steps:\t{sample_steps}")
        print(f"\tCFG scale:\t\t{cfg_scale}")
        print(f"\tSeed:\t\t{seed}")
        print(f"\tOutput format:\t{fmt}")
        print("=== Output ===")
        print(f"\tTotal prompts:\t{len(conds.keys())}")
        print(f"\tItems per prompt:\t{n_sample_per_cond}")
        print('')

    path_rank = path_full[rank:: world_size]
    conds_rank = conds_full[rank:: world_size]

    # Generation
    model.eval()
    n_iter = int(math.ceil(len(conds_rank) / batch_sample))
    for idx in range(n_iter):
        path_i = path_rank[idx * batch_sample: (idx + 1) * batch_sample]
        conds_i = conds_rank[idx * batch_sample: (idx + 1) * batch_sample]

        samples_i = generate_diffusion_cond(
            model,
            steps=sample_steps,
            cfg_scale=cfg_scale,
            conditioning=conds_i,
            sample_size=sample_size,
            seed=seed,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type=sampler_type,
            device=device,
            disable_tqdm=(rank != 0)
        )

        for idx_n in range(samples_i.shape[0]):
            audio = float_to_int16_audio(samples_i[idx_n])
            if clip_length:
                L = int(conds_i[idx_n]['seconds_total'] * sample_rate)
                audio = audio[:, :L]
            save_path = f"{output_dir}/{path_i[idx_n]}.{fmt}"
            save_audio(audio, save_path, sample_rate, fmt)

    print(f"->->-> Rank-{rank}: Finished.")


if __name__ == "__main__":
    main()
