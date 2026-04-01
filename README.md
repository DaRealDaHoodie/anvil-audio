# Anvil Audio

> **A pluggable studio tool for AI audio generation. Swap models, keep your workflow.**

Anvil Audio is a refactored and extended fork of
[`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) by Stability AI.
It turns a single-model inference codebase into a clean, swappable-component platform where
models, conditioners, and compressors are first-class abstractions.

Supports **Stable Audio** diffusion models and **ACE-Step** music-generation models through
a unified registry, CLI, and Gradio UI.

---

## What's New in Anvil

- **Pluggable pipeline architecture** — `BasePipeline`, `BaseGenerator`, `BaseCompressor`, `BaseConditioner` ABCs; swap any component without touching the rest of your workflow.
- **Named model registry** — `anvil generate --model stable-audio-open-1.0 --prompt "..."` loads the right pipeline automatically; add your own entries in `~/.anvil-audio/registry.yaml`.
- **ACE-Step support** — optional integration with [ACE-Step v1.5](https://github.com/ace-step/ACE-Step) for full-song music generation with lyrics and style tags.
- **Output management** — collision-free timestamped filenames, JSON metadata sidecars, batch manifests, and project-scoped folders under `~/anvil-audio-outputs/`.
- **MPS / CUDA / CPU auto-detection** — runs on Apple Silicon, NVIDIA GPUs, or CPU with no flags needed.
- **`anvil generate` CLI** — multi-GPU via Accelerate, wav/flac/mp3 output, batch YAML conditions, per-run seed control.
- **Gradio web UI** — project name, seed input, live metadata panel, model dropdown with hot-reload, device field.
- **Built-in audio editor** — post-processing tab with normalize, trim, fade, time stretch, pitch shift, EQ, and reverb; non-destructive exports with full effects sidecar.
- **MCP server** — expose all generation and editing capabilities to Claude and other MCP clients over stdio; models are cached between calls.
- **Python 3.12+** — uses modern union syntax, `slots=True` dataclasses, and lowercase generics throughout.

---

## Requirements

- Python **3.12 or 3.13** (strongly recommended — Python 3.14 is too new for several ML dependencies and will cause build failures)
- PyTorch 2.0 or later

---

## Install

> **Python 3.12 or 3.13 is strongly recommended.** Python 3.14 is too new — several ML
> dependencies (scipy, k-diffusion) don't have pre-built wheels for it yet and will attempt
> to compile from source. If you're on Homebrew Python, check your version with
> `python3 --version` and install 3.12/3.13 via `brew install python@3.13` if needed.

```bash
git clone https://github.com/DaRealDaHoodie/anvil-audio.git
cd anvil-audio

# Python 3.12+ requires a virtual environment (mandatory on Homebrew / system Python)
python3.13 -m venv .venv        # use python3.12 if 3.13 isn't available
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install .
# avoid Accelerate import error on some setups
pip uninstall -y transformer-engine
```

**If you must use Python 3.14** and hit scipy build errors, install a Fortran compiler first:

```bash
brew install gcc        # provides gfortran, required to compile scipy from source
pip install .
```

---

## Quick Start

The fastest path to generating audio:

```bash
# 1. Clone and install (see Install above)
# 2. Add a model to your registry (see User Registry below)
# 3. Launch the Gradio UI — loads your first registered model automatically
python run_gradio.py
```

Or from the CLI in one command:

```bash
anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"
```

---

## Stable Audio Models

### CLI

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
# Load by registry name (recommended)
python run_gradio.py --model stable-audio-open-1.0

# Load from HuggingFace Hub directly
python run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0

# No args — loads the first model from your registry
python run_gradio.py

# Route outputs to a named project folder
python run_gradio.py --model stable-audio-open-1.0 --project sfx-pack-v1
```

---

## ACE-Step Music Generation (optional)

[ACE-Step](https://github.com/ace-step/ACE-Step) is an open-source full-song music generation
model that supports style tags and full lyric input. Anvil integrates it through the same
registry and UI as Stable Audio — no separate server or app required.

ACE-Step is **optional**. If you don't install it, all other Anvil functionality works as normal.

### Install ACE-Step

```bash
# Clone the ACE-Step repo (Anvil imports it directly — no pip install needed)
git clone https://github.com/ace-step/ACE-Step.git ~/Developer/GitRepos/ACE-Step-1.5
cd ~/Developer/GitRepos/ACE-Step-1.5

# Install ACE-Step's dependencies into your Anvil venv
pip install -r requirements.txt

# ACE-Step requires transformers 4.x (not 5.x)
pip install "transformers>=4.51.0,<4.58.0"
```

The model weights are downloaded automatically from HuggingFace the first time you generate.

To use a custom install path, set `ACESTEP_PROJECT_ROOT` before launching:

```bash
export ACESTEP_PROJECT_ROOT=/path/to/your/ACE-Step
```

### Built-in registry entries

Anvil registers two ACE-Step variants automatically when the repo is found:

| Model name | Description | Steps | Notes |
|---|---|---|---|
| `acestep-v1.5-turbo` | Fast generation | 50 | Good for drafts and quick iteration |
| `acestep-v1.5-sft` | Full quality | 100 | Better for final exports |

### CLI

```bash
# Single prompt (instrumental)
anvil generate --model acestep-v1.5-turbo \
    --prompt "indie pop, acoustic guitar, warm vocals, upbeat"

# Batch generation with lyrics (see example file)
anvil generate --model acestep-v1.5-turbo \
    --cond-yaml-path example/generation/acestep_conditions.yaml \
    --output-dir ./out
```

Batch YAML format — each entry supports `prompt`, `lyrics`, and `seconds_total`:

```yaml
tracks:
  indie_pop:
    prompt: "indie pop, acoustic guitar, warm vocals, upbeat, sunny afternoon"
    lyrics: |
      [verse]
      Walking down the open road
      Sunlight through the trees
      [chorus]
      This is where I start again
    seconds_total: 30.0

  electronic_instrumental:
    prompt: "electronic, synthwave, driving bass, retro 80s, cinematic"
    lyrics: "[Instrumental]"
    seconds_total: 45.0
```

### Gradio web UI

```bash
# Turbo (fast, good for iteration)
python run_gradio.py --model acestep-v1.5-turbo

# Full quality
python run_gradio.py --model acestep-v1.5-sft
```

The ACE-Step UI adds a **Lyrics** field below the prompt. Leave it blank or enter
`[Instrumental]` for tracks with no vocals. Structure lyrics with section markers
like `[verse]`, `[chorus]`, `[bridge]`.

---

## Audio Editor

The Gradio UI includes a built-in **Edit** tab for quick post-processing without
leaving Anvil. It works with any loaded model and any audio file — not just
files generated in the current session.

Available tools:

| Tool | What it does |
|---|---|
| Normalize | Peak or LUFS loudness targeting |
| Trim silence | Strips quiet sections from the edges |
| Fade in / fade out | Linear ramp from/to silence |
| Loop / clip | Trim the audio to a start/end range |
| Time stretch | Speed up or slow down without changing pitch |
| Pitch shift | Transpose up or down in semitones |
| EQ | Low shelf, peak mid, high shelf |
| Reverb | Room size, damping, wet/dry mix |

Effects are applied in a fixed chain (trim → clip → stretch/pitch → EQ →
reverb → fade → normalize), which keeps results predictable regardless of the
order you adjust knobs.

**Typical workflow:**

1. Generate on the **Generate** tab
2. Switch to **Edit**
3. Click **Load Last Generation** — the output loads automatically
4. Adjust effects; click **Preview** to hear the result
5. Click **Export** when satisfied

Export creates a new file via the output manager — the original is never
touched. The JSON sidecar for the exported file records the source path and
the full effects chain so you can always trace what was applied and replay it.

You can also drag any audio file into the source field to edit files from
outside Anvil.

---

## MCP Server

Anvil exposes its full capabilities as an [MCP](https://modelcontextprotocol.io) server so
Claude and other MCP clients can generate and edit audio directly without the Gradio UI or
manual CLI commands.

### Install

```bash
pip install mcp
```

The `mcp` package is not installed by default. Everything else is already a dependency.

### Available tools

| Tool | What it does |
|---|---|
| `generate_audio` | Generate a clip from a prompt; auto-selects model if not specified |
| `batch_generate` | Generate multiple clips in one call |
| `edit_audio` | Post-process a file with normalize, trim, EQ, reverb, etc. |
| `list_models` | All registered models with type, limits, and loaded status |
| `get_model_info` | Full details for one model |
| `list_recent_outputs` | Recent output files with their metadata, newest-first |
| `get_generation_metadata` | Read the sidecar for any output file |
| `list_projects` | Project folders under `~/anvil-audio-outputs/` |
| `set_active_project` | Set a default project so you don't repeat it every call |

Models are loaded lazily on first use and cached between calls — switching between
two models during a session only pays the load cost once per model.

### Claude Desktop config

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`
(create the file if it doesn't exist):

```json
{
  "mcpServers": {
    "anvil-audio": {
      "command": "/path/to/anvil-audio/.venv/bin/python",
      "args": ["-m", "anvil_audio.mcp_server"]
    }
  }
}
```

Replace `/path/to/anvil-audio` with the absolute path to your clone.

### Claude Code config

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "anvil-audio": {
      "command": "/path/to/anvil-audio/.venv/bin/python",
      "args": ["-m", "anvil_audio.mcp_server"],
      "type": "stdio"
    }
  }
}
```

### Example session

Once configured, Claude can generate and edit audio directly:

```
You:    Generate a short thunderstorm ambience clip
Claude: [calls generate_audio(prompt="thunderstorm ambience, rain, distant thunder", duration_seconds=20)]
        Generated: ~/anvil-audio-outputs/default/20260401_181907_thunderstorm_...wav

You:    Add a slight fade in and normalize it to -14 LUFS
Claude: [calls edit_audio(file_path="...", fade_in=2.0, normalize=True,
                          normalize_target_db=-14, normalize_lufs=True)]
        Exported: ~/anvil-audio-outputs/default/20260401_181942_edit_...wav
```

---

## User Registry

Add your own models to `~/.anvil-audio/registry.yaml`. The file is a YAML list — entries
with the same name as a built-in will override it.

### Stable Audio / diffusion models

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

### ACE-Step models

```yaml
- name: my-acestep-finetune
  pipeline_type: acestep
  acestep_project_root: /path/to/ACE-Step    # path to the cloned repo
  model_config_path: acestep-v15-turbo        # checkpoint variant name
  default_params:
    steps: 50
    cfg_scale: 4.0
    audio_duration: 60
```

---

## `run_gradio.py` flags

| Flag | Description |
|------|-------------|
| `--model` | Registry model name (e.g. `stable-audio-open-1.0`, `acestep-v1.5-turbo`) |
| `--pretrained-name` | HuggingFace Hub repo ID (e.g. `stabilityai/stable-audio-open-1.0`) |
| `--model-config` | Local model config JSON (ignored if `--model` or `--pretrained-name` set) |
| `--ckpt-path` | Local checkpoint (ignored if `--model` or `--pretrained-name` set) |
| `--pretransform-ckpt-path` | Optional separate VAE checkpoint |
| `--username` / `--password` | Gradio auth |
| `--model-half` | Use float16 inference |
| `--device` | `cuda`, `mps`, or `cpu` (auto-detects if omitted) |
| `--project` | Outputs go to `~/anvil-audio-outputs/{project}/` |
| `--share` | Create a public Gradio share URL |

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
| `--sample-steps` | pipeline default | Diffusion / inference steps |
| `--cfg-scale` | pipeline default | CFG guidance scale |
| `--sampler-type` | pipeline default | Sampler type |
| `--sigma-min` / `--sigma-max` | pipeline default | Noise schedule bounds |
| `--n-sample-per-cond` | `1` | Samples per condition |
| `--batch-size` | `10` | Items per GPU batch |
| `--seed` | `-1` (random) | RNG seed |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

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

## Todo

- [x] Pluggable pipeline architecture (BasePipeline, BaseGenerator, etc.)
- [x] Named model registry with user-extensible YAML
- [x] Output manager (collision-free filenames, JSON sidecars, project folders)
- [x] `anvil generate` CLI with multi-GPU, batch YAML, seed control
- [x] Gradio UI with model hot-reload and live metadata
- [x] ACE-Step v1.5 integration (turbo + SFT, lyrics support, Gradio UI)
- [ ] Post-processing pipeline (EQ, normalize, fade)
- [ ] PyPI package (`pip install anvil-audio`)
- [x] MCP server integration
- [ ] Add more audio augmentations
- [ ] Add troubleshooting section
- [ ] Add contribution guidelines

---

## Licensing

Anvil Audio is MIT licensed. It builds on several open-source projects and optional model
weights with their own licenses — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
full attributions.
