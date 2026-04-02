"""
ACEStepPipeline — BasePipeline adapter for ACE-Step v1.5.

ACE-Step is kept as a separate package (pip-installable or local source tree).
This module imports from ``acestep`` at construction time only; a missing
installation raises a clear ``ImportError`` with remediation instructions
rather than a cryptic ``ModuleNotFoundError`` at import time.

Vocabulary mapping
------------------
Anvil conditioning key  →  ACE-Step parameter
``prompt``              →  ``captions`` (style / genre tags)
``lyrics``              →  ``lyrics`` (vocal content; use ``"[Instrumental]"``
                           or leave blank for instrumental output)
``seconds_total``       →  ``audio_duration`` (generation length in seconds)
``seed``                →  passed as the ``seed`` argument
``steps``               →  ``inference_steps``
``cfg_scale``           →  ``guidance_scale``
``scheduler_type``      →  ``infer_method`` (``"ode"`` or ``"sde"``)

Usage
-----
::

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    pipe = ACEStepPipeline(
        project_root="/path/to/ACE-Step-1.5",
        config_path="acestep-v15-turbo",
        device="auto",
    )
    audio = pipe.generate(
        [{"prompt": "upbeat indie pop", "lyrics": "[verse]\\nHello world", "seconds_total": 30}]
    )
    # audio: Tensor [1, 2, T] at 48 kHz
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from anvil_audio.core.interfaces import BasePipeline
from anvil_audio.utils.stdio_guard import stdout_to_stderr


class ACEStepPipeline(BasePipeline):
    """``BasePipeline`` adapter for ACE-Step v1.5.

    Wraps ``AceStepHandler`` so ACE-Step integrates with Anvil's registry,
    CLI batch generation, ``OutputManager``, and Gradio UI without copying
    any of ACE-Step's model weights or inference logic.

    Args:
        project_root:    Absolute path to the ACE-Step repository root (the
                         directory that contains ``checkpoints/``).  Can be
                         a cloned git repo or an installed package directory.
        config_path:     Model variant — ``"acestep-v15-turbo"`` (8 steps,
                         fast) or ``"acestep-v15-sft"`` (full quality, 32+
                         steps).  Defaults to ``"acestep-v15-turbo"``.
        device:          Device hint passed to ``initialize_service`` — one
                         of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.
                         ``"auto"`` lets ACE-Step pick the best available.
        offload_to_cpu:  Enable sequential CPU offloading to lower peak VRAM
                         usage at the cost of slower generation.
        default_params:  Generation parameter overrides applied when callers
                         omit individual kwargs.  Recognised keys:
                         ``steps``, ``cfg_scale``, ``scheduler_type``,
                         ``sigma_min``, ``sigma_max`` (last two stored as 0.0
                         so ``GenerationMetadata`` serialises cleanly).
    """

    #: ACE-Step v1.5 VAE outputs 48 kHz stereo audio.
    _SAMPLE_RATE: int = 48000

    def __init__(
        self,
        project_root: str,
        config_path: str = "acestep-v15-turbo",
        device: str = "auto",
        offload_to_cpu: bool = False,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        # Resolve and (if needed) inject the project root so that
        # `from acestep.handler import ...` works whether ACE-Step is
        # pip-installed or run directly from its cloned source tree.
        resolved_root = str(Path(project_root).resolve())
        if resolved_root not in sys.path:
            sys.path.insert(0, resolved_root)

        try:
            from acestep.handler import AceStepHandler  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ACE-Step could not be imported.  Either install it via\n\n"
                "    pip install ace_step\n\n"
                "or point 'project_root' at the cloned ACE-Step repository "
                "so that its source tree is importable.\n\n"
                f"Underlying error: {exc}"
            ) from exc

        self._handler: Any = AceStepHandler()
        self._project_root: str = resolved_root
        self._config_path: str = config_path
        self._device_str: str = device

        self.default_params: dict[str, Any] = default_params or {
            "steps": 8,
            "cfg_scale": 7.0,
            "scheduler_type": "ode",
            # sigma_min / sigma_max are Stable-Audio concepts; store 0.0 so
            # GenerationMetadata can serialise them without KeyError.
            "sigma_min": 0.0,
            "sigma_max": 0.0,
        }

        print(
            f"->->-> Initialising ACE-Step  "
            f"config={config_path!r}  device={device!r}  "
            f"offload={offload_to_cpu}",
            file=sys.stderr,
        )
        # initialize_service loads model weights and may print to stdout;
        # redirect so MCP stdio is not corrupted.
        with stdout_to_stderr():
            status, success = self._handler.initialize_service(
                project_root=self._project_root,
                config_path=config_path,
                device=device,
                offload_to_cpu=offload_to_cpu,
            )
        if not success:
            raise RuntimeError(
                f"ACE-Step model initialisation failed.\n"
                f"  project_root : {self._project_root}\n"
                f"  config_path  : {config_path!r}\n"
                f"  device       : {device!r}\n"
                f"  Status       : {status}"
            )
        print(f"->->-> ACE-Step ready  {status}", file=sys.stderr)

    # ------------------------------------------------------------------
    # BasePipeline abstract property implementations
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """48 kHz — native output sample rate of ACE-Step v1.5."""
        return self._SAMPLE_RATE

    @property
    def sample_size(self) -> int:
        """Default sample count for a 30-second clip (48 000 × 30)."""
        return self._SAMPLE_RATE * 30

    # ------------------------------------------------------------------
    # BasePipeline abstract method implementations
    # ------------------------------------------------------------------

    def generate(
        self,
        conditioning: list[dict[str, Any]],
        steps: int | None = None,
        seed: int = -1,
        **kwargs: Any,
    ) -> Tensor:
        """Generate a batch of stereo music waveforms via ACE-Step.

        Each condition dict may contain:

        ============  =====================================================
        Key           Description
        ============  =====================================================
        ``prompt``    Tags / style caption (e.g. ``"upbeat indie pop"``).
        ``lyrics``    Lyric text.  Omit or pass ``"[Instrumental]"`` for
                      instrumental output.
        ``seconds_total`` Target duration in seconds.  ``None`` or ``≤0``
                      lets ACE-Step choose automatically.
        ============  =====================================================

        Args:
            conditioning: List of B condition dicts.
            steps:        Diffusion steps.  Falls back to
                          ``default_params["steps"]`` (8 for turbo).
            seed:         RNG seed; -1 draws a random seed each call.
            **kwargs:     Per-call overrides:
                          - ``cfg_scale`` (float): guidance strength
                          - ``scheduler_type`` (str): ``"ode"`` or ``"sde"``

        Returns:
            Float32 tensor ``[B, 2, T]`` in ``[-1, 1]`` at 48 kHz.

        Raises:
            RuntimeError: If ACE-Step returns a failure payload.
        """
        effective_steps = (
            steps if steps is not None else self.default_params.get("steps", 8)
        )
        effective_cfg = float(
            kwargs.get("cfg_scale", self.default_params.get("cfg_scale", 7.0))
        )
        effective_method = str(
            kwargs.get("scheduler_type", self.default_params.get("scheduler_type", "ode"))
        )
        use_random_seed = seed == -1

        audio_tensors: list[Tensor] = []

        for cond in conditioning:
            tags: str = cond.get("prompt", "")
            lyrics: str = cond.get("lyrics", "")

            raw_dur = cond.get("seconds_total")
            duration: float | None = (
                float(raw_dur) if isinstance(raw_dur, (int, float)) and float(raw_dur) > 0
                else None
            )

            # generate_music may print progress to stdout; redirect so MCP
            # stdio is not corrupted.
            with stdout_to_stderr():
                result = self._handler.generate_music(
                    captions=tags,
                    lyrics=lyrics,
                    inference_steps=int(effective_steps),
                    guidance_scale=effective_cfg,
                    seed=seed,
                    use_random_seed=use_random_seed,
                    audio_duration=duration,
                    infer_method=effective_method,
                    batch_size=1,
                )

            if not result.get("success", False):
                raise RuntimeError(
                    f"ACE-Step generation failed: "
                    f"{result.get('error', 'unknown error')}"
                )

            audios: list[dict[str, Any]] = result.get("audios", [])
            if not audios:
                raise RuntimeError(
                    "ACE-Step returned an empty audio list; "
                    "check the handler logs for details."
                )

            # audios[0] = {"tensor": Tensor[C, T], "sample_rate": int}
            audio_t: Tensor = audios[0]["tensor"].float()
            if audio_t.dim() == 1:
                # Mono: unsqueeze to [1, T]
                audio_t = audio_t.unsqueeze(0)

            audio_tensors.append(audio_t.cpu())

        # Pad shorter clips to the batch maximum length, then stack → [B, C, T]
        max_len = max(t.shape[-1] for t in audio_tensors)
        padded = [
            F.pad(t, (0, max_len - t.shape[-1])) if t.shape[-1] < max_len else t
            for t in audio_tensors
        ]
        return torch.stack(padded)

    def to(self, device: str | torch.device) -> "ACEStepPipeline":
        """No-op: ACE-Step initialises its device at construction time.

        ACE-Step does not support moving weights post-initialisation.  This
        method is present for ``BasePipeline`` interface compatibility and
        returns ``self`` without modification.
        """
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def eval(self) -> "ACEStepPipeline":
        """No-op (ACE-Step is always in eval mode).  Returns ``self``."""
        return self

    def __repr__(self) -> str:
        return (
            f"ACEStepPipeline("
            f"config={self._config_path!r}, "
            f"device={self._device_str!r}, "
            f"sample_rate={self.sample_rate}"
            f")"
        )
