"""
Model registry for anvil_audio.

Provides a central catalogue of named pipeline configurations.  Each entry
maps a short name (e.g. ``"stable-audio-open-1.0"``) to everything needed to
instantiate a ready-to-run ``BasePipeline``.

Built-in entries cover known Stability AI HuggingFace models.  Users can
extend or override the registry via a YAML file at::

    ~/.anvil-audio/registry.yaml

YAML schema (all fields optional except ``name``)::

    - name: my-sfx-model
      pretrained_name: myorg/my-sfx-model      # HuggingFace Hub repo
      model_config_path: /path/to/config.json  # local config (alternative)
      ckpt_path: /path/to/model.ckpt           # local checkpoint
      default_params:
        steps: 100
        cfg_scale: 7.0
        sampler_type: dpmpp-3m-sde
        sigma_min: 0.3
        sigma_max: 500

Usage::

    from anvil_audio.core import registry, load_pipeline

    pipeline = load_pipeline("stable-audio-open-1.0")
    audio = pipeline.generate([{"prompt": "rain on a tin roof", "seconds_start": 0, "seconds_total": 10}])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import DiffusionPipeline


# ---------------------------------------------------------------------------
# Default generation parameters
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[str, Any] = {
    "steps": 100,
    "cfg_scale": 7.0,
    "sampler_type": "dpmpp-3m-sde",
    "sigma_min": 0.3,
    "sigma_max": 500.0,
}

# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegistryEntry:
    """A single named model configuration.

    At least one of ``pretrained_name`` or (``model_config_path`` +
    ``ckpt_path``) must be provided.

    Attributes:
        name:               Short identifier used as the ``--model`` CLI flag.
        pretrained_name:    HuggingFace Hub repository ID.
        model_config_path:  Path to a local JSON model config file.
        ckpt_path:          Path to a local model checkpoint (.ckpt or .safetensors).
        pretransform_ckpt_path: Optional path to a separate pretransform checkpoint.
        default_params:     Generation parameters merged over global defaults.
                            Recognised keys: steps, cfg_scale, sampler_type,
                            sigma_min, sigma_max.
    """

    name: str
    pretrained_name: str | None = None
    model_config_path: str | None = None
    ckpt_path: str | None = None
    pretransform_ckpt_path: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict)

    def resolved_params(self) -> dict[str, Any]:
        """Return generation params merged over the global defaults."""
        return {**_DEFAULT_PARAMS, **self.default_params}

    def validate(self) -> None:
        """Raise ``ValueError`` if the entry is not usable."""
        has_hf = self.pretrained_name is not None
        has_local = self.model_config_path is not None and self.ckpt_path is not None
        if not has_hf and not has_local:
            raise ValueError(
                f"Registry entry '{self.name}' must have either 'pretrained_name' "
                "or both 'model_config_path' and 'ckpt_path'."
            )


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Central catalogue of named pipeline configurations.

    Thread-safety: registration is not thread-safe; load *before* spawning
    workers.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._load_builtin()
        self._load_user_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, entry: RegistryEntry) -> None:
        """Add or replace an entry in the registry.

        Args:
            entry: A ``RegistryEntry`` instance.  Existing entries with the
                   same name are silently replaced.
        """
        entry.validate()
        self._entries[entry.name] = entry

    def get(self, name: str) -> RegistryEntry:
        """Return the entry for *name*, raising ``KeyError`` if not found.

        Args:
            name: The model name as registered (case-sensitive).

        Raises:
            KeyError: If no entry with that name exists.
        """
        if name not in self._entries:
            available = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(
                f"Model '{name}' not found in registry.  "
                f"Available: {available}"
            )
        return self._entries[name]

    def get_model(self, name: str) -> "RegistryEntry | None":
        """Return the entry for *name*, or ``None`` if not found."""
        return self._entries.get(name)

    def list_models(self) -> list[RegistryEntry]:
        """Return all registered entries sorted by name."""
        return sorted(self._entries.values(), key=lambda e: e.name)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_builtin(self) -> None:
        """Register known Stability AI / HuggingFace models."""
        builtin: list[RegistryEntry] = [
            RegistryEntry(
                name="stable-audio-open-1.0",
                pretrained_name="stabilityai/stable-audio-open-1.0",
                default_params={
                    "steps": 100,
                    "cfg_scale": 7.0,
                    "sampler_type": "dpmpp-3m-sde",
                    "sigma_min": 0.3,
                    "sigma_max": 500.0,
                },
            ),
            RegistryEntry(
                name="stable-audio-open-small",
                pretrained_name="stabilityai/stable-audio-open-small",
                default_params={
                    "steps": 25,
                    "cfg_scale": 1.0,
                    "sampler_type": "dpmpp-3m-sde",
                    "sigma_min": 0.3,
                    "sigma_max": 1.0,
                },
            ),
        ]
        for entry in builtin:
            self._entries[entry.name] = entry

    def _load_user_registry(self) -> None:
        """Merge entries from ``~/.anvil-audio/registry.yaml`` if it exists.

        User entries override built-in entries with the same name.
        Invalid entries are skipped with a printed warning.
        """
        registry_path = Path.home() / ".anvil-audio" / "registry.yaml"
        if not registry_path.exists():
            return

        try:
            import yaml  # optional at import time; required only when file exists
        except ImportError:
            print(
                f"[registry] WARNING: {registry_path} found but PyYAML is not "
                "installed.  Run `pip install pyyaml` to enable user registry."
            )
            return

        try:
            with registry_path.open() as fh:
                raw = yaml.safe_load(fh)
        except Exception as exc:
            print(f"[registry] WARNING: Could not read {registry_path}: {exc}")
            return

        if not isinstance(raw, list):
            print(
                f"[registry] WARNING: {registry_path} must contain a YAML list "
                "of model entries.  Skipping."
            )
            return

        for item in raw:
            if not isinstance(item, dict) or "name" not in item:
                print(f"[registry] WARNING: Skipping invalid entry: {item!r}")
                continue
            try:
                entry = RegistryEntry(
                    name=item["name"],
                    pretrained_name=item.get("pretrained_name"),
                    model_config_path=item.get("model_config_path"),
                    ckpt_path=item.get("ckpt_path"),
                    pretransform_ckpt_path=item.get("pretransform_ckpt_path"),
                    default_params=item.get("default_params") or {},
                )
                entry.validate()
                self._entries[entry.name] = entry
            except (ValueError, TypeError) as exc:
                print(f"[registry] WARNING: Skipping entry '{item.get('name')}': {exc}")


# ---------------------------------------------------------------------------
# Module-level singleton and convenience helpers
# ---------------------------------------------------------------------------

#: Global registry instance — import this directly::
#:
#:     from anvil_audio.core import registry
#:     registry.register(RegistryEntry(name="my-model", ...))
registry = ModelRegistry()


def load_pipeline(
    name: str,
    device: str | None = None,
) -> "DiffusionPipeline":
    """Load a ``DiffusionPipeline`` by registry name.

    This is the primary high-level entry point for programmatic use.

    Args:
        name:   Registry name (e.g. ``"stable-audio-open-1.0"``).
        device: Target device string (``"cuda"``, ``"mps"``, ``"cpu"``).
                Auto-detects via ``get_best_device()`` if omitted.

    Returns:
        A ``DiffusionPipeline`` with the model loaded and moved to *device*.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    # Import here to avoid circular imports at module load time
    from .pipeline import DiffusionPipeline
    from anvil_audio.utils.torch_common import get_best_device

    entry = registry.get(name)
    _device = device or str(get_best_device())

    if entry.pretrained_name is not None:
        from anvil_audio.models.pretrained import get_pretrained_model
        model, model_config = get_pretrained_model(entry.pretrained_name)
    else:
        import json
        from anvil_audio.models.factory import create_model_from_config
        from anvil_audio.models.utils import load_ckpt_state_dict
        from anvil_audio.utils.torch_common import copy_state_dict
        with open(entry.model_config_path) as fh:
            model_config = json.load(fh)
        model = create_model_from_config(model_config)
        copy_state_dict(model, load_ckpt_state_dict(entry.ckpt_path))

    pipeline = DiffusionPipeline(
        model=model,
        model_config=model_config,
        default_params=entry.resolved_params(),
    )
    pipeline.to(_device)
    return pipeline
