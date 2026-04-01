# Anvil Audio

> **A pluggable studio tool for AI audio generation. Swap models, keep your workflow.**

Anvil Audio is a refactored and extended fork of
[`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) by Stability AI.
It turns a single-model inference codebase into a clean, swappable-component platform where
models, conditioners, and compressors are first-class abstractions.

---

## What's New in Anvil

- **Pluggable pipeline architecture** — `BasePipeline`, `BaseGenerator`, `BaseCompressor`, `BaseConditioner` ABCs; swap any component without touching the rest of your workflow.
- **Named model registry** — `anvil generate --model stable-audio-open-1.0 --prompt "..."` loads the right pipeline automatically; add your own entries in `~/.anvil-audio/registry.yaml`.
- **Output management** — collision-free timestamped filenames, JSON metadata sidecars, batch manifests, and project-scoped folders under `~/anvil-audio-outputs/`.
- **MPS / CUDA / CPU auto-detection** — runs on Apple Silicon, NVIDIA GPUs, or CPU with no flags needed.
- **`anvil generate` CLI** — multi-GPU via Accelerate, wav/flac/mp3 output, batch YAML conditions, per-run seed control.
- **Gradio web UI** — project name, seed input, live metadata panel, model dropdown with hot-reload, device field.
- **Python 3.12+** — uses modern union syntax, `slots=True` dataclasses, and lowercase generics throughout.

---

## Credits

Built on top of [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) (MIT) by Stability AI
and the [`friendly-stable-audio-tools`](https://github.com/yukara-ikemiya/friendly-stable-audio-tools) refactor by Yukara Ikemiya.
The Stable Audio model family remains the work of Stability AI.

---

## Requirements

- Python 3.12 or later
- PyTorch 2.0 or later (for Flash Attention support)

---

## Install

```bash
git clone https://github.com/DaRealDaHoodie/anvil-audio.git
cd anvil-audio
pip install .
# avoid Accelerate import error on some setups
pip uninstall -y transformer-engine
```

---

## Quick Start

### Generate from the CLI

```bash
# Use a registered model by name
anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"

# List all registered models
anvil generate --list-models

# Batch generation from a YAML file
anvil generate --model stable-audio-open-1.0 --cond-yaml-path batch.yaml --output-dir ./out

# Legacy path (local config + checkpoint)
anvil generate --model-config config.json --ckpt-path model.ckpt \
    --prompt "rain on a tin roof" --output-dir ./out
```

Multi-GPU generation is supported via Accelerate.

### Gradio web UI

```bash
# Pretrained model from Hugging Face Hub
python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0

# Apple Silicon
python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0 --device mps

# Route outputs to a named project folder
python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0 --project sfx-pack-v1
```

`run_gradio.py` flags:

| Flag | Description |
|------|-------------|
| `--pretrained-name` | HuggingFace Hub repo ID (e.g. `stabilityai/stable-audio-open-1.0`) |
| `--model-config` | Local model config JSON (ignored if `--pretrained-name` set) |
| `--ckpt-path` | Local checkpoint (ignored if `--pretrained-name` set) |
| `--pretransform-ckpt-path` | Optional separate VAE checkpoint |
| `--username` / `--password` | Gradio auth |
| `--model-half` | Use float16 inference |
| `--device` | `cuda`, `mps`, or `cpu` (auto-detects if omitted) |
| `--project` | Outputs go to `~/anvil-audio-outputs/{project}/` |

---

## User Registry

Add your own models to `~/.anvil-audio/registry.yaml`:

```yaml
- name: my-sfx-model
  pretrained_name: myorg/my-sfx-model        # HuggingFace Hub
  default_params:
    steps: 100
    cfg_scale: 7.0

- name: local-vae-dit
  model_config_path: /path/to/config.json
  ckpt_path: /path/to/model.ckpt
  pretransform_ckpt_path: /path/to/vae.ckpt
```

---

## Logging

Training requires a [Weights & Biases](https://wandb.ai) account:

```bash
wandb login
# or pass as env var
export WANDB_API_KEY="your-key-here"
```

---

## Training

### Configuration files

You need two config files before starting a training run:

- **model config** — defines architecture and training hyperparameters
- **dataset config** — points to your audio and metadata

See [docs/datasets.md](docs/datasets.md) for dataset config details.

### Training from scratch

```bash
python3 train.py \
    --dataset-config /path/to/dataset/config \
    --model-config /path/to/model/config \
    --name my_experiment
```

### Fine-tuning

- Resume from a wrapped checkpoint: `--ckpt-path path/to/wrapped.ckpt`
- Start fresh from an unwrapped pre-trained model: `--pretrained-ckpt-path path/to/unwrapped.ckpt`

### Unwrapping a model

Training checkpoints include the full training wrapper (discriminators, EMA, optimizer states).
Unwrap before using for inference or as a pretransform:

```bash
python3 unwrap_model.py \
    --model-config /path/to/model/config \
    --ckpt-path /path/to/wrapped/ckpt.ckpt \
    --name /path/to/output/unwrapped_name
```

---

## Training Stable Audio 2.0

### Prerequisites

**1. CLAP encoder checkpoint**

Download `music_audioset_epoch_15_esc_90.14.pt` from the
[LAION CLAP repository](https://github.com/LAION-AI/CLAP?tab=readme-ov-file#pretrained-models)
and set `clap_ckpt_path` in `stable_audio_2_0.json`:

```json
"config": {
    "clap_ckpt_path": "ckpt/clap/music_audioset_epoch_15_esc_90.14.pt"
}
```

**2. Audio + metadata**

Each audio file needs a paired JSON sidecar with at minimum a `prompt` field:

```
dataset/
├── music_1.wav
├── music_1.json   ← {"prompt": "upbeat electronic track with positive vibes"}
├── music_2.wav
├── music_2.json
└── ...
```

### Stage 1 — VAE-GAN

```bash
MODEL_CONFIG="anvil_audio/configs/model_configs/autoencoders/stable_audio_2_0_vae.json"
DATASET_CONFIG="anvil_audio/configs/dataset_configs/local_training_example.json"

python3 train.py \
    --dataset-config ${DATASET_CONFIG} \
    --model-config ${MODEL_CONFIG} \
    --name "vae_training" \
    --num-gpus 8 \
    --batch-size 10 \
    --num-workers 8 \
    --save-dir ./output
```

After training, unwrap the checkpoint before Stage 2.

### Stage 2 — Diffusion Transformer (DiT)

```bash
MODEL_CONFIG="anvil_audio/configs/model_configs/txt2audio/stable_audio_2_0.json"
PRETRANSFORM_CKPT="/path/to/unwrapped_vae.ckpt"

python3 train.py \
    --dataset-config ${DATASET_CONFIG} \
    --model-config ${MODEL_CONFIG} \
    --pretransform-ckpt-path ${PRETRANSFORM_CKPT} \
    --name "dit_training" \
    --num-gpus 8 \
    --batch-size 10 \
    --save-dir ./output
```

### Reconstruction test

```bash
python3 reconstruct_audios.py \
    --model-config ${MODEL_CONFIG} \
    --ckpt-path /path/to/unwrapped_vae.ckpt \
    --audio-dir /path/to/original_audio/ \
    --output-dir /path/to/reconstructed/ \
    --frame-duration 1.0 \
    --overlap-rate 0.01 \
    --batch-size 50
```

---

## Container Setup

Build a Docker image and optionally convert to Singularity for HPC clusters:

```bash
NAME=anvil-audio
docker build -t ${NAME} -f ./container/friendly-stable-audio-tools.Dockerfile .

# Convert to Singularity
singularity build anvil-audio.sif docker-daemon://anvil-audio
```

---

## `anvil generate` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model NAME` | — | Registry model name |
| `--list-models` | — | Print registry and exit |
| `--model-config PATH` | — | Legacy: local JSON config |
| `--ckpt-path PATH` | — | Legacy: local checkpoint |
| `--pretransform-ckpt-path PATH` | — | Separate VAE checkpoint |
| `--prompt TEXT` | — | Single text prompt |
| `--cond-yaml-path PATH` | — | Batch YAML conditions file |
| `--seconds-start` | `0.0` | Start time (seconds) |
| `--seconds-total` | `30.0` | Duration (seconds) |
| `--output-dir` | `./output` | Output directory |
| `--format` | `wav` | `wav`, `flac`, or `mp3` |
| `--clip-length` | off | Clip to `seconds_total` |
| `--sample-steps` | pipeline default | Diffusion steps |
| `--cfg-scale` | pipeline default | CFG guidance scale |
| `--sampler-type` | pipeline default | Sampler type |
| `--sigma-min` / `--sigma-max` | pipeline default | Noise schedule bounds |
| `--n-sample-per-cond` | `1` | Samples per condition |
| `--batch-size` | `10` | Items per GPU batch |
| `--seed` | `-1` (random) | RNG seed |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

---

## Todo

- [ ] Add more audio augmentations
- [ ] Add troubleshooting section
- [ ] Add contribution guidelines
- [ ] MCP server integration
