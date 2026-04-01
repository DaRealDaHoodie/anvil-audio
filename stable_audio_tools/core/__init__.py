"""
stable_audio_tools.core
~~~~~~~~~~~~~~~~~~~~~~~
Pluggable component abstraction layer for generative audio models.

Public surface:
    BaseCompressor, BaseGenerator, BaseConditioner, BasePipeline  — ABCs
    ModelRegistry, RegistryEntry, registry                        — model registry
    load_pipeline                                                  — convenience loader
"""

from .interfaces import BaseCompressor, BaseGenerator, BaseConditioner, BasePipeline
from .registry import ModelRegistry, RegistryEntry, registry, load_pipeline

__all__ = [
    "BaseCompressor",
    "BaseGenerator",
    "BaseConditioner",
    "BasePipeline",
    "ModelRegistry",
    "RegistryEntry",
    "registry",
    "load_pipeline",
]
