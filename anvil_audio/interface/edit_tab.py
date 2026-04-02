"""
Post-processing Edit tab for the Gradio UI.

Processing chain (applied in fixed order):
  1. Trim silence        — librosa.effects.trim
  2. Loop / clip         — trim to [start_s, end_s]
  3. Time stretch + pitch shift — pedalboard.time_stretch
  4. EQ                  — pedalboard LowShelfFilter + PeakFilter + HighShelfFilter
  5. Reverb              — pedalboard.Reverb
  6. Fade in / fade out  — numpy linear ramp
  7. Normalize           — peak or LUFS (pyloudnorm)

Non-destructive: the source audio is never modified in place.
Export saves through OutputManager with a JSON sidecar containing the source
path, the source sidecar path (if it exists), and all effect parameters.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

def _to_pb(sr: int, audio: np.ndarray) -> tuple[int, np.ndarray]:
    """Gradio (sr, (N,) | (N,C) int16) → (sr, (C,N) float32) for pedalboard."""
    arr = audio.astype(np.float32) / 32768.0
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]     # mono → (1, N)
    else:
        arr = arr.T                  # (N, C) → (C, N)
    return sr, arr


def _from_pb(sr: int, audio: np.ndarray) -> tuple[int, np.ndarray]:
    """(sr, (C,N) float32) → Gradio (sr, (N,) | (N,C) int16)."""
    arr = np.clip(audio, -1.0, 1.0)
    arr = (arr * 32767.0).astype(np.int16)
    if arr.shape[0] == 1:
        return sr, arr[0]            # mono → (N,)
    return sr, arr.T                 # (C, N) → (N, C)


# ---------------------------------------------------------------------------
# Processing chain
# ---------------------------------------------------------------------------

def process_audio_chain(
    audio_input: tuple[int, np.ndarray] | None,
    # 1. Trim silence
    trim_enabled: bool,
    trim_top_db: float,
    # 2. Loop / clip
    loop_enabled: bool,
    loop_start_s: float,
    loop_end_s: float,
    # 3. Time stretch + pitch shift
    stretch_factor: float,
    pitch_semitones: float,
    # 4. EQ
    eq_enabled: bool,
    low_gain_db: float,
    low_freq: float,
    mid_gain_db: float,
    mid_freq: float,
    high_gain_db: float,
    high_freq: float,
    # 5. Reverb
    reverb_enabled: bool,
    reverb_room_size: float,
    reverb_damping: float,
    reverb_wet_dry: float,
    # 6. Fade
    fade_in_s: float,
    fade_out_s: float,
    # 7. Normalize
    normalize_enabled: bool,
    normalize_target_db: float,
    normalize_use_lufs: bool,
) -> tuple[int, np.ndarray] | None:
    """Run the full processing chain and return Gradio-format audio or None."""
    if audio_input is None:
        return None

    in_sr, in_arr = audio_input
    if in_arr is None or in_arr.size == 0:
        return None

    sr, arr = _to_pb(in_sr, in_arr)   # arr: (C, N) float32

    # ------------------------------------------------------------------
    # 1. Trim silence
    # ------------------------------------------------------------------
    if trim_enabled and arr.shape[1] > 0:
        try:
            import librosa
            ref = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
            _, idx = librosa.effects.trim(ref.astype(np.float32), top_db=float(trim_top_db))
            start, end = int(idx[0]), int(idx[1])
            if end > start:
                arr = arr[:, start:end]
        except Exception as exc:
            print(f"[edit] Trim failed: {exc}")

    # ------------------------------------------------------------------
    # 2. Loop / clip to range
    # ------------------------------------------------------------------
    if loop_enabled and arr.shape[1] > 0:
        n = arr.shape[1]
        s = max(0, min(int(loop_start_s * sr), n))
        e = max(s, min(int(loop_end_s * sr), n))
        if e > s:
            arr = arr[:, s:e]

    # ------------------------------------------------------------------
    # 3. Time stretch + pitch shift
    # ------------------------------------------------------------------
    needs_stretch = abs(stretch_factor - 1.0) > 1e-4
    needs_pitch = abs(pitch_semitones) > 1e-4
    if (needs_stretch or needs_pitch) and arr.shape[1] > 0:
        try:
            import pedalboard
            arr = pedalboard.time_stretch(
                arr,
                sr,
                stretch_factor=float(stretch_factor),
                pitch_shift_in_semitones=float(pitch_semitones),
            )
        except Exception as exc:
            print(f"[edit] Time stretch failed: {exc}")

    # ------------------------------------------------------------------
    # 4. EQ
    # ------------------------------------------------------------------
    if eq_enabled and arr.shape[1] > 0:
        try:
            import pedalboard
            plugins: list[Any] = []
            if abs(low_gain_db) > 0.05:
                plugins.append(pedalboard.LowShelfFilter(
                    cutoff_frequency_hz=float(low_freq),
                    gain_db=float(low_gain_db),
                ))
            if abs(mid_gain_db) > 0.05:
                plugins.append(pedalboard.PeakFilter(
                    cutoff_frequency_hz=float(mid_freq),
                    gain_db=float(mid_gain_db),
                    q=1.0,
                ))
            if abs(high_gain_db) > 0.05:
                plugins.append(pedalboard.HighShelfFilter(
                    cutoff_frequency_hz=float(high_freq),
                    gain_db=float(high_gain_db),
                ))
            if plugins:
                board = pedalboard.Pedalboard(plugins)
                arr = board(arr, sr)
        except Exception as exc:
            print(f"[edit] EQ failed: {exc}")

    # ------------------------------------------------------------------
    # 5. Reverb
    # ------------------------------------------------------------------
    if reverb_enabled and reverb_wet_dry > 1e-4 and arr.shape[1] > 0:
        try:
            import pedalboard
            wet = float(reverb_wet_dry)
            dry = 1.0 - wet
            board = pedalboard.Pedalboard([
                pedalboard.Reverb(
                    room_size=float(reverb_room_size),
                    damping=float(reverb_damping),
                    wet_level=wet,
                    dry_level=dry,
                )
            ])
            arr = board(arr, sr)
        except Exception as exc:
            print(f"[edit] Reverb failed: {exc}")

    # ------------------------------------------------------------------
    # 6. Fade in / fade out
    # ------------------------------------------------------------------
    n = arr.shape[1]
    if fade_in_s > 1e-4 and n > 0:
        fade_samp = min(int(fade_in_s * sr), n)
        ramp = np.linspace(0.0, 1.0, fade_samp, dtype=np.float32)
        arr[:, :fade_samp] *= ramp

    if fade_out_s > 1e-4 and n > 0:
        fade_samp = min(int(fade_out_s * sr), n)
        ramp = np.linspace(1.0, 0.0, fade_samp, dtype=np.float32)
        arr[:, n - fade_samp:] *= ramp

    # ------------------------------------------------------------------
    # 7. Normalize
    # ------------------------------------------------------------------
    if normalize_enabled and arr.shape[1] > 0:
        if normalize_use_lufs:
            try:
                import pyloudnorm as pyln
                data = arr.T.astype(np.float64)   # (N, C) float64
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(data)
                if np.isfinite(loudness) and loudness > -70.0:
                    normalized = pyln.normalize.loudness(
                        data, loudness, float(normalize_target_db)
                    )
                    arr = normalized.T.astype(np.float32)
            except Exception as exc:
                print(f"[edit] LUFS normalize failed: {exc}")
        else:
            peak = float(np.max(np.abs(arr)))
            if peak > 1e-8:
                target_linear = 10.0 ** (float(normalize_target_db) / 20.0)
                arr = arr * (target_linear / peak)

    return _from_pb(sr, arr)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_audio(
    audio_input: tuple[int, np.ndarray] | None,
    source_path: str,
    project: str,
    # chain params — same order as process_audio_chain
    trim_enabled: bool,
    trim_top_db: float,
    loop_enabled: bool,
    loop_start_s: float,
    loop_end_s: float,
    stretch_factor: float,
    pitch_semitones: float,
    eq_enabled: bool,
    low_gain_db: float,
    low_freq: float,
    mid_gain_db: float,
    mid_freq: float,
    high_gain_db: float,
    high_freq: float,
    reverb_enabled: bool,
    reverb_room_size: float,
    reverb_damping: float,
    reverb_wet_dry: float,
    fade_in_s: float,
    fade_out_s: float,
    normalize_enabled: bool,
    normalize_target_db: float,
    normalize_use_lufs: bool,
    export_fmt: str = "wav",
) -> str:
    """Run the chain and save via OutputManager. Returns a status string."""
    result = process_audio_chain(
        audio_input,
        trim_enabled, trim_top_db,
        loop_enabled, loop_start_s, loop_end_s,
        stretch_factor, pitch_semitones,
        eq_enabled, low_gain_db, low_freq, mid_gain_db, mid_freq, high_gain_db, high_freq,
        reverb_enabled, reverb_room_size, reverb_damping, reverb_wet_dry,
        fade_in_s, fade_out_s,
        normalize_enabled, normalize_target_db, normalize_use_lufs,
    )
    if result is None:
        return "Error: no audio to export. Upload or load a file first."

    out_sr, out_arr = result

    # Convert back to (C, N) float32 tensor for OutputManager
    import torch
    float_arr = out_arr.astype(np.float32) / 32768.0
    if float_arr.ndim == 1:
        float_arr = float_arr[np.newaxis, :]
    else:
        float_arr = float_arr.T   # (N, C) → (C, N)
    tensor = torch.from_numpy(float_arr)

    from ..core.output import GenerationMetadata, OutputManager

    source_p = source_path.strip() if source_path else ""
    source_sidecar = ""
    if source_p:
        candidate = Path(source_p).with_suffix(".json")
        if candidate.exists():
            source_sidecar = str(candidate)

    effects_config = {
        "trim": {"enabled": trim_enabled, "top_db": trim_top_db},
        "loop": {"enabled": loop_enabled, "start_s": loop_start_s, "end_s": loop_end_s},
        "time_stretch": {"factor": stretch_factor, "pitch_semitones": pitch_semitones},
        "eq": {
            "enabled": eq_enabled,
            "low": {"gain_db": low_gain_db, "freq": low_freq},
            "mid": {"gain_db": mid_gain_db, "freq": mid_freq},
            "high": {"gain_db": high_gain_db, "freq": high_freq},
        },
        "reverb": {
            "enabled": reverb_enabled,
            "room_size": reverb_room_size,
            "damping": reverb_damping,
            "wet_dry": reverb_wet_dry,
        },
        "fade": {"in_s": fade_in_s, "out_s": fade_out_s},
        "normalize": {
            "enabled": normalize_enabled,
            "target_db": normalize_target_db,
            "mode": "lufs" if normalize_use_lufs else "peak",
        },
    }

    fmt = export_fmt if export_fmt in ("wav", "mp3", "ogg", "flac") else "wav"

    meta = GenerationMetadata(
        prompt="(edit)",
        model_name="edit",
        seed=-1,
        steps=0,
        cfg_scale=1.0,
        sampler_type="edit",
        sigma_min=0.0,
        sigma_max=0.0,
        duration_seconds=float_arr.shape[1] / out_sr,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        extra={
            "source_audio_path": source_p,
            "source_sidecar_path": source_sidecar,
            "export_format": fmt,
            "effects": effects_config,
        },
    )

    output_manager = OutputManager(project=project or None)
    path, _ = output_manager.save_audio(tensor, meta, out_sr, ext=fmt)
    return f"Exported: {path}"


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def _load_audio_from_path(path: str) -> tuple[int, np.ndarray] | None:
    """Load a WAV/FLAC from disk and return Gradio-format (sr, int16 ndarray)."""
    if not path:
        return None
    try:
        import soundfile as sf
        # sf.read returns (N,) mono or (N, C) stereo in float64, already Gradio layout
        data, sr = sf.read(path, dtype="int16", always_2d=False)
        return sr, data
    except Exception as exc:
        print(f"[edit] Failed to load {path}: {exc}")
        return None


def create_edit_tab(
    project_component: Any,
    last_path_getter: Callable[[], str],
) -> None:
    """Build the 'Edit' gr.Tab content.

    Args:
        project_component: Shared ``gr.Textbox`` for the project name.
        last_path_getter:  Zero-argument callable returning the path of the
                           most recently generated audio file (may be ``""``).
    """
    import gradio as gr

    # ------------------------------------------------------------------
    # Source audio row
    # ------------------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=5):
            audio_input = gr.Audio(
                label="Source audio — upload a file or click 'Load Last Generation'",
                type="numpy",
                interactive=True,
            )
        with gr.Column(scale=1):
            load_last_btn = gr.Button("Load Last Generation", variant="secondary")
            source_path_state = gr.State("")

    # ------------------------------------------------------------------
    # Effect controls  (left) + preview player (right)
    # ------------------------------------------------------------------
    with gr.Row(equal_height=False):
        with gr.Column():

            with gr.Accordion("Trim Silence", open=False):
                trim_enabled = gr.Checkbox(label="Enable", value=False)
                trim_top_db = gr.Slider(
                    minimum=10, maximum=80, step=1, value=40,
                    label="Threshold (dB below peak)",
                    info="Audio quieter than this threshold at the edges is trimmed. Lower = more aggressive trimming.",
                )

            with gr.Accordion("Loop / Clip", open=False):
                loop_enabled = gr.Checkbox(label="Enable", value=False)
                with gr.Row():
                    loop_start_s = gr.Slider(
                        minimum=0.0, maximum=600.0, step=0.01, value=0.0,
                        label="Start (seconds)",
                        info="Trim audio before this point.",
                    )
                    loop_end_s = gr.Slider(
                        minimum=0.0, maximum=600.0, step=0.01, value=30.0,
                        label="End (seconds)",
                        info="Trim audio after this point.",
                    )

            with gr.Accordion("Time Stretch & Pitch Shift", open=False):
                with gr.Row():
                    stretch_factor = gr.Slider(
                        minimum=0.25, maximum=4.0, step=0.01, value=1.0,
                        label="Stretch factor",
                        info="1.0 = no change. 0.5 = half speed (2× longer). 2.0 = double speed (half as long). Pitch is kept constant unless you also change the pitch slider.",
                    )
                    pitch_semitones = gr.Slider(
                        minimum=-24.0, maximum=24.0, step=0.5, value=0.0,
                        label="Pitch shift (semitones)",
                        info="Shifts pitch without affecting tempo. ±12 = one octave. Processed simultaneously with stretch so only one pass of pitch shifting occurs.",
                    )

            with gr.Accordion("EQ", open=False):
                eq_enabled = gr.Checkbox(label="Enable", value=False)
                with gr.Row():
                    low_gain_db = gr.Slider(
                        minimum=-24.0, maximum=24.0, step=0.5, value=0.0,
                        label="Low shelf gain (dB)",
                    )
                    low_freq = gr.Slider(
                        minimum=20.0, maximum=800.0, step=10.0, value=200.0,
                        label="Low shelf frequency (Hz)",
                    )
                with gr.Row():
                    mid_gain_db = gr.Slider(
                        minimum=-24.0, maximum=24.0, step=0.5, value=0.0,
                        label="Mid peak gain (dB)",
                    )
                    mid_freq = gr.Slider(
                        minimum=200.0, maximum=8000.0, step=50.0, value=1000.0,
                        label="Mid peak frequency (Hz)",
                    )
                with gr.Row():
                    high_gain_db = gr.Slider(
                        minimum=-24.0, maximum=24.0, step=0.5, value=0.0,
                        label="High shelf gain (dB)",
                    )
                    high_freq = gr.Slider(
                        minimum=2000.0, maximum=20000.0, step=500.0, value=8000.0,
                        label="High shelf frequency (Hz)",
                    )

            with gr.Accordion("Reverb", open=False):
                reverb_enabled = gr.Checkbox(label="Enable", value=False)
                with gr.Row():
                    reverb_room_size = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.25,
                        label="Room size",
                        info="Controls the size of the simulated room. Larger = longer reverb tail.",
                    )
                    reverb_damping = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                        label="Damping",
                        info="High-frequency energy absorption. Higher = darker, shorter reverb tail.",
                    )
                    reverb_wet_dry = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.25,
                        label="Wet/dry mix",
                        info="0 = fully dry (no reverb), 1 = fully wet (reverb only).",
                    )

            with gr.Accordion("Fade", open=False):
                with gr.Row():
                    fade_in_s = gr.Slider(
                        minimum=0.0, maximum=10.0, step=0.01, value=0.0,
                        label="Fade in (seconds)",
                        info="Linear ramp from silence at the start of the file.",
                    )
                    fade_out_s = gr.Slider(
                        minimum=0.0, maximum=10.0, step=0.01, value=0.0,
                        label="Fade out (seconds)",
                        info="Linear ramp to silence at the end of the file.",
                    )

            with gr.Accordion("Normalize", open=False):
                normalize_enabled = gr.Checkbox(label="Enable", value=False)
                with gr.Row():
                    normalize_target_db = gr.Slider(
                        minimum=-30.0, maximum=0.0, step=0.5, value=-1.0,
                        label="Target level (dBFS / LUFS)",
                        info="For peak mode: target peak amplitude. For LUFS mode: target integrated loudness.",
                    )
                    normalize_use_lufs = gr.Checkbox(
                        label="LUFS mode (vs peak)",
                        value=False,
                        info="Peak normalizes the loudest sample. LUFS targets perceptual loudness (-14 LUFS is streaming standard).",
                    )

        with gr.Column():
            preview_audio = gr.Audio(
                label="Preview — click Preview to hear the result before exporting",
                interactive=False,
            )
            with gr.Row():
                preview_btn = gr.Button("Preview", variant="primary")
                export_fmt_dropdown = gr.Dropdown(
                    ["wav", "mp3", "ogg"],
                    value="wav",
                    label="Format",
                    min_width=80,
                    info="wav = lossless · mp3 = compressed · ogg = game engines (Roblox, Unity, Godot)",
                )
                export_btn = gr.Button("Export", variant="secondary")
            export_status = gr.Textbox(
                label="Export status",
                interactive=False,
            )

    # ------------------------------------------------------------------
    # Collect chain param components in the fixed chain order
    # ------------------------------------------------------------------
    _chain_inputs = [
        audio_input,
        trim_enabled, trim_top_db,
        loop_enabled, loop_start_s, loop_end_s,
        stretch_factor, pitch_semitones,
        eq_enabled, low_gain_db, low_freq, mid_gain_db, mid_freq, high_gain_db, high_freq,
        reverb_enabled, reverb_room_size, reverb_damping, reverb_wet_dry,
        fade_in_s, fade_out_s,
        normalize_enabled, normalize_target_db, normalize_use_lufs,
    ]

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _load_last() -> tuple[Any, str]:
        path = last_path_getter()
        audio = _load_audio_from_path(path)
        return audio, path

    load_last_btn.click(
        fn=_load_last,
        inputs=[],
        outputs=[audio_input, source_path_state],
    )

    preview_btn.click(
        fn=process_audio_chain,
        inputs=_chain_inputs,
        outputs=[preview_audio],
    )

    export_btn.click(
        fn=export_audio,
        inputs=[audio_input, source_path_state, project_component] + _chain_inputs[1:] + [export_fmt_dropdown],
        outputs=[export_status],
    )
