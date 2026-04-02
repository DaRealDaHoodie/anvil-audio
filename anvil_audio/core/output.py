"""
Output management — filename generation, sidecar metadata, batch manifests.

Pure library code: no dependency on the CLI, Gradio, or any ML runtime.
Importable and callable from an external process (e.g. an MCP server)
without launching the Gradio UI or the generate.py CLI::

    from anvil_audio.core.output import OutputManager, GenerationMetadata
    manager = OutputManager(project="sfx-pack-v1")

Output root layout::

    ~/anvil-audio-outputs/
    └── {project}/                          ← single-item files land here
        ├── 20240115_143022_rain_on_tin_roof_stable_audio_open_1_0_42.wav
        ├── 20240115_143022_rain_on_tin_roof_stable_audio_open_1_0_42.json
        └── 20240115_143100_batch_4items/   ← batch subfolder
            ├── batch_manifest.json
            ├── 20240115_143100_door_creak_...wav
            └── ...
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor


# ---------------------------------------------------------------------------
# GenerationMetadata
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GenerationMetadata:
    """All parameters that produced a given audio file.

    Serialisable to/from JSON via ``to_dict()`` / ``from_dict()``.  The
    ``output_path`` field is set by ``OutputManager.save_audio()`` after the
    file is written — callers should leave it as the default empty string.

    Attributes:
        prompt:           Text prompt used for generation.
        model_name:       Registry name or ``"custom"`` for local configs.
        seed:             The actual RNG seed used (never -1; always resolved
                          before generation so the file is reproducible).
        steps:            Number of diffusion / flow steps.
        cfg_scale:        Classifier-free guidance scale.
        sampler_type:     Sampler identifier (e.g. ``"dpmpp-3m-sde"``).
        sigma_min:        Minimum sigma for the noise schedule.
        sigma_max:        Maximum sigma for the noise schedule.
        duration_seconds: Actual duration of the saved audio clip.
        timestamp:        Generation start time as ``"YYYYMMDD_HHMMSS"``.
        output_path:      Absolute path to the saved audio file (set post-save).
        negative_prompt:  Negative conditioning text, if any.
        seconds_start:    Model-level start-time conditioning.
        seconds_total:    Model-level total-duration conditioning.
        generation_duration_seconds: Wall-clock time from start of inference to
                          audio file written, in seconds.  ``None`` when not
                          measured (e.g. legacy sidecars or non-generation ops).
        extra:            Catch-all for pipeline-specific params.
    """

    prompt: str
    model_name: str
    seed: int
    steps: int
    cfg_scale: float
    sampler_type: str
    sigma_min: float
    sigma_max: float
    duration_seconds: float
    timestamp: str
    output_path: str = ""
    negative_prompt: str = ""
    seconds_start: float = 0.0
    seconds_total: float = 0.0
    generation_duration_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of all fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationMetadata":
        """Reconstruct from a dict (e.g. parsed from a sidecar file).

        Unknown keys are folded into the ``extra`` field.
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra_in = {k: v for k, v in d.items() if k not in known}
        filtered = {k: v for k, v in d.items() if k in known}
        if extra_in:
            existing_extra = filtered.get("extra") or {}
            filtered["extra"] = {**extra_in, **existing_extra}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GenerationResult:
    """The output of one generation item: saved file + metadata.

    ``audio`` is the raw float32 waveform ``[C, T]`` in ``[-1, 1]`` as it
    came out of the pipeline.  Keep a reference if post-processing is needed
    (e.g. spectrograms); otherwise it can be discarded once the file is saved.

    Attributes:
        audio:        Float waveform tensor ``[C, T]``.
        path:         Absolute path to the saved audio file.
        sidecar_path: Absolute path to the saved ``.json`` sidecar.
        metadata:     Full generation metadata.
    """

    audio: "Tensor"
    path: Path
    sidecar_path: Path
    metadata: GenerationMetadata


# ---------------------------------------------------------------------------
# OutputManager
# ---------------------------------------------------------------------------

class OutputManager:
    """Routes generated files to project directories, names them consistently,
    writes JSON sidecars, and prevents overwrites.

    Filename format::

        {YYYYMMDD_HHMMSS}_{prompt_slug}_{model_slug}_{seed}.{ext}

    where ``prompt_slug`` and ``model_slug`` are sanitised to
    alphanumeric + underscore, max 40 chars each.

    Batch subfolder::

        {YYYYMMDD_HHMMSS}_batch_{N}items/   ← contains audio + sidecar pairs
            batch_manifest.json              ← index of all items

    Args:
        project:  Subdirectory name inside *base_dir*.  Defaults to
                  ``"default"``.  Whitespace-only values are treated as
                  ``"default"``.
        base_dir: Override the root directory.  Defaults to
                  ``~/anvil-audio-outputs``.
    """

    DEFAULT_BASE: Path = Path.home() / "anvil-audio-outputs"

    def __init__(
        self,
        project: str | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        base = Path(base_dir) if base_dir else self.DEFAULT_BASE
        proj = project.strip() if project and project.strip() else "default"
        self.project: str = proj
        self.output_root: Path = base / proj
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_slug(text: str, max_len: int = 40) -> str:
        """Convert arbitrary text to a safe filename slug.

        Lowercases input, replaces runs of non-alphanumeric characters with
        ``_``, strips leading/trailing underscores, truncates to *max_len*,
        and returns ``"untitled"`` for blank/empty input.

        Args:
            text:    Input string (prompt, model name, etc.).
            max_len: Maximum character length of the output slug.

        Returns:
            A lowercase alphanumeric-plus-underscore string.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
        slug = slug.strip("_")[:max_len].rstrip("_")
        return slug or "untitled"

    @staticmethod
    def no_overwrite(path: Path) -> Path:
        """Return *path* if it does not exist, else ``stem_001``, ``stem_002``…

        Args:
            path: Desired output path.

        Returns:
            A path that does not yet exist on disk.

        Raises:
            RuntimeError: After 9999 attempts (should never happen in practice).
        """
        if not path.exists():
            return path
        stem, suffix, parent = path.stem, path.suffix, path.parent
        for i in range(1, 10_000):
            candidate = parent / f"{stem}_{i:03d}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not find a free filename near {path}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def build_filename(
        self,
        metadata: GenerationMetadata,
        ext: str,
        ts_override: str | None = None,
    ) -> str:
        """Construct the filename string (no directory component).

        Args:
            metadata:    Source of prompt, model name, seed, timestamp.
            ext:         File extension without dot (``"wav"``, ``"flac"``…).
            ts_override: If provided, use this timestamp instead of
                         ``metadata.timestamp``.

        Returns:
            A filename string like
            ``"20240115_143022_wooden_door_creak_sfx_v1_42.wav"``.
        """
        ts = ts_override or metadata.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        p_slug = self.sanitize_slug(metadata.prompt)
        m_slug = self.sanitize_slug(metadata.model_name)
        return f"{ts}_{p_slug}_{m_slug}_{metadata.seed}.{ext}"

    def resolve_path(
        self,
        metadata: GenerationMetadata,
        ext: str = "wav",
        subfolder: Path | None = None,
    ) -> Path:
        """Return a unique, collision-free output path.

        Args:
            metadata:  Used for filename construction.
            ext:       File extension (without dot).
            subfolder: If given, place the file there; otherwise use
                       ``self.output_root``.

        Returns:
            An absolute ``Path`` that does not yet exist.
        """
        parent = subfolder if subfolder is not None else self.output_root
        fname = self.build_filename(metadata, ext)
        return self.no_overwrite(parent / fname)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def write_sidecar(
        self,
        audio_path: Path,
        metadata: GenerationMetadata,
    ) -> Path:
        """Write a ``.json`` sidecar adjacent to *audio_path*.

        The sidecar is placed at ``audio_path.with_suffix(".json")``.
        ``metadata.output_path`` is set to the string form of *audio_path*
        before serialisation so the sidecar is self-contained.

        Args:
            audio_path: Path of the audio file that was (or will be) saved.
            metadata:   Metadata to serialise.

        Returns:
            Path to the written sidecar file.
        """
        data = metadata.to_dict()
        data["output_path"] = str(audio_path)
        sidecar = audio_path.with_suffix(".json")
        sidecar.write_text(json.dumps(data, indent=2, default=str))
        return sidecar

    def save_audio(
        self,
        audio: "Tensor",
        metadata: GenerationMetadata,
        sample_rate: int,
        ext: str = "wav",
        subfolder: Path | None = None,
    ) -> tuple[Path, Path]:
        """Save *audio* to disk and write a JSON sidecar.

        Args:
            audio:       Float waveform tensor ``[C, T]`` in ``[-1, 1]``.
            metadata:    Generation metadata.  ``output_path`` is mutated to
                         reflect the final saved path.
            sample_rate: Audio sample rate in Hz.
            ext:         Output format extension (``"wav"``, ``"flac"``,
                         ``"mp3"``).
            subfolder:   Optional directory override (for batch runs).

        Returns:
            ``(audio_path, sidecar_path)`` — both absolute ``Path`` objects.
        """
        from anvil_audio.utils.audio_utils import float_to_int16_audio, save_audio

        path = self.resolve_path(metadata, ext, subfolder)
        path.parent.mkdir(parents=True, exist_ok=True)

        audio_int16 = audio if audio.dtype == torch.int16 else float_to_int16_audio(audio)
        save_audio(str(path), audio_int16, sample_rate, fmt=ext)

        metadata.output_path = str(path)
        sidecar = self.write_sidecar(path, metadata)
        return path, sidecar

    def write_batch_manifest(
        self,
        batch_dir: Path,
        results: list[GenerationResult],
    ) -> Path:
        """Write ``batch_manifest.json`` to *batch_dir*.

        Args:
            batch_dir: Directory where all batch audio files live.
            results:   Ordered list of ``GenerationResult`` items.

        Returns:
            Path to the written manifest file.
        """
        manifest: dict[str, Any] = {
            "batch_dir": str(batch_dir),
            "item_count": len(results),
            "items": [
                {
                    "path": str(r.path),
                    "sidecar": str(r.sidecar_path),
                    "metadata": r.metadata.to_dict(),
                }
                for r in results
            ],
        }
        manifest_path = batch_dir / "batch_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        return manifest_path

    def make_batch_dir(self, n_items: int) -> Path:
        """Create and return a fresh timestamped subfolder for a batch run.

        Args:
            n_items: Item count, embedded in the folder name for clarity.

        Returns:
            The created ``Path``.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.output_root / f"{ts}_batch_{n_items}items"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # ------------------------------------------------------------------
    # Static reader
    # ------------------------------------------------------------------

    @staticmethod
    def read_sidecar(path: Path | str) -> GenerationMetadata:
        """Read and return ``GenerationMetadata`` from a sidecar ``.json`` file.

        If *path* has an audio extension, the corresponding ``.json`` sidecar
        is derived automatically.

        Args:
            path: Path to the ``.json`` sidecar or to the audio file.

        Returns:
            ``GenerationMetadata`` instance.

        Raises:
            FileNotFoundError: If the sidecar file does not exist.
        """
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        if not p.exists():
            raise FileNotFoundError(f"Sidecar not found: {p}")
        return GenerationMetadata.from_dict(json.loads(p.read_text()))
