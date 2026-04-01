"""
Abstract base classes that define the component contracts for swappable
audio generation components.

Every concrete model plugged into this framework must implement one of:
    BaseCompressor   — audio ↔ latent codec
    BaseGenerator    — latent-space denoising / generation backbone
    BaseConditioner  — raw input → embedding encoder
    BasePipeline     — end-to-end orchestrator

Design notes
------------
* BaseCompressor, BaseGenerator, and BaseConditioner extend nn.Module so
  they participate in PyTorch's device/dtype management normally.
* BasePipeline is a pure Python ABC (not an nn.Module).  It is an
  orchestrator that *holds* modules; callers use its .to() method to move
  everything at once.
* All abstract properties are declared with @property + @abstractmethod so
  subclasses must provide them as computed or plain attributes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# BaseCompressor
# ---------------------------------------------------------------------------

class BaseCompressor(nn.Module, ABC):
    """Interface for any audio autoencoder / codec.

    A compressor is responsible for two symmetric operations:
        encode : waveform  → latent representation
        decode : latents   → waveform

    Subclasses must expose the three shape properties so that callers can
    reason about tensor dimensions without instantiating the model.
    """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def downsampling_ratio(self) -> int:
        """Temporal compression factor.

        ``encode(audio).shape[-1] == audio.shape[-1] // downsampling_ratio``
        """

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Number of channels in the latent tensor."""

    @property
    @abstractmethod
    def io_channels(self) -> int:
        """Number of audio channels expected by ``encode`` (e.g. 1 or 2)."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, audio: Tensor) -> Tensor:
        """Encode a batch of waveforms to latents.

        Args:
            audio: Float tensor of shape ``[B, C, T]`` where ``C`` matches
                   ``self.io_channels``.

        Returns:
            Latent tensor of shape ``[B, latent_dim, T // downsampling_ratio]``.
        """

    @abstractmethod
    def decode(self, latents: Tensor) -> Tensor:
        """Decode a batch of latents back to waveforms.

        Args:
            latents: Float tensor of shape ``[B, latent_dim, L]``.

        Returns:
            Waveform tensor of shape ``[B, io_channels, L * downsampling_ratio]``.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def latent_length(self, audio_length: int) -> int:
        """Return the latent time dimension for a given number of audio samples."""
        return audio_length // self.downsampling_ratio


# ---------------------------------------------------------------------------
# BaseGenerator
# ---------------------------------------------------------------------------

class BaseGenerator(nn.Module, ABC):
    """Interface for any latent-space generative model.

    A generator takes a noise tensor and a dict of pre-encoded conditioning
    tensors and iteratively denoises them into a clean latent.  The caller
    is responsible for sampling from the prior (noise generation) and for
    decoding the output latent back to audio via a ``BaseCompressor``.

    The ``conditioning`` dict is intentionally untyped — different generators
    accept different keys.  Concrete implementations should document the
    keys they consume.
    """

    @property
    @abstractmethod
    def io_channels(self) -> int:
        """Number of latent channels the generator operates on."""

    @abstractmethod
    def generate(
        self,
        noise: Tensor,
        conditioning: dict[str, Any],
        steps: int,
        **kwargs: Any,
    ) -> Tensor:
        """Run the generative process.

        Args:
            noise:        Initial noise tensor ``[B, io_channels, L]``.
            conditioning: Dict of pre-encoded conditioning tensors (e.g.
                          ``{"cross_attn_cond": ..., "global_cond": ...}``).
            steps:        Number of denoising steps.
            **kwargs:     Sampler-specific options (cfg_scale, sigma_min, …).

        Returns:
            Denoised latent tensor of the same shape as ``noise``.
        """


# ---------------------------------------------------------------------------
# BaseConditioner
# ---------------------------------------------------------------------------

class BaseConditioner(nn.Module, ABC):
    """Interface for any conditioning encoder.

    A conditioner maps raw inputs (strings, audio clips, numbers, …) to a
    fixed-width embedding tensor plus an attention mask.  The output contract
    matches the existing ``Conditioner.forward`` contract so adapters are
    one-liners.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by ``encode``."""

    @abstractmethod
    def encode(self, inputs: list[Any]) -> tuple[Tensor, Tensor]:
        """Encode a batch of raw inputs to embeddings.

        Args:
            inputs: List of B raw inputs (strings, tensors, ints, …).

        Returns:
            A 2-tuple ``(embeddings, mask)`` where:
                embeddings : ``[B, seq_len, output_dim]``
                mask       : ``[B, seq_len]`` boolean or float attention mask
        """


# ---------------------------------------------------------------------------
# BasePipeline
# ---------------------------------------------------------------------------

class BasePipeline(ABC):
    """Orchestrator that wires a compressor, generator, and conditioner(s)
    into a complete end-to-end generation flow.

    ``BasePipeline`` is *not* an ``nn.Module``; it is a façade that holds
    modules internally and exposes a single high-level ``generate`` method.
    Device management is handled explicitly via ``to()``.

    Subclasses own the details of:
        * how conditioning inputs are pre-processed
        * noise initialisation
        * sampler dispatch
        * latent → audio decoding
    """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate produced by this pipeline (Hz)."""

    @property
    @abstractmethod
    def sample_size(self) -> int:
        """Default number of audio samples generated per call."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        conditioning: list[dict[str, Any]],
        steps: int = 100,
        seed: int = -1,
        **kwargs: Any,
    ) -> Tensor:
        """Generate a batch of audio waveforms.

        Args:
            conditioning: List of B condition dicts, each containing keys
                          appropriate for this pipeline (e.g. ``"prompt"``,
                          ``"seconds_start"``, ``"seconds_total"``).
            steps:        Number of diffusion/flow steps.
            seed:         RNG seed; -1 means random.
            **kwargs:     Pipeline-specific options forwarded to the sampler
                          (cfg_scale, sampler_type, sigma_min, sigma_max, …).

        Returns:
            Float audio tensor of shape ``[B, channels, T]`` in ``[-1, 1]``.
        """

    @abstractmethod
    def to(self, device: str | torch.device) -> "BasePipeline":
        """Move all internal modules to *device* and return ``self``."""

    # ------------------------------------------------------------------
    # Optional component accessors
    # ------------------------------------------------------------------

    def get_compressor(self) -> "BaseCompressor | None":
        """Return the pipeline's compressor component, if any."""
        return None

    def get_generator(self) -> "BaseGenerator | None":
        """Return the pipeline's generator component, if any."""
        return None
