from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


class _FakeGenerationParams:
    def __init__(
        self,
        *,
        task_type: str,
        caption: str,
        lyrics: str,
        vocal_language: str,
        bpm: int | None,
        keyscale: str,
        timesignature: str,
        duration: float,
        inference_steps: int,
        guidance_scale: float,
        infer_method: str,
        shift: float,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        thinking: bool,
        lm_temperature: float,
        lm_cfg_scale: float,
        lm_top_k: int,
        lm_top_p: float,
        lm_negative_prompt: str,
        use_cot_metas: bool,
        use_cot_caption: bool,
        use_cot_language: bool,
        use_constrained_decoding: bool,
    ) -> None:
        kwargs = {
            "task_type": task_type,
            "caption": caption,
            "lyrics": lyrics,
            "vocal_language": vocal_language,
            "bpm": bpm,
            "keyscale": keyscale,
            "timesignature": timesignature,
            "duration": duration,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "infer_method": infer_method,
            "shift": shift,
            "use_adg": use_adg,
            "cfg_interval_start": cfg_interval_start,
            "cfg_interval_end": cfg_interval_end,
            "thinking": thinking,
            "lm_temperature": lm_temperature,
            "lm_cfg_scale": lm_cfg_scale,
            "lm_top_k": lm_top_k,
            "lm_top_p": lm_top_p,
            "lm_negative_prompt": lm_negative_prompt,
            "use_cot_metas": use_cot_metas,
            "use_cot_caption": use_cot_caption,
            "use_cot_language": use_cot_language,
            "use_constrained_decoding": use_constrained_decoding,
        }
        self.__dict__.update(kwargs)


class _FakeGenerationConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def test_acestep_pipeline_uses_upstream_generation_params(monkeypatch, tmp_path):
    calls: dict[str, Any] = {}

    class FakeAceStepHandler:
        def initialize_service(self, **kwargs: Any) -> tuple[str, bool]:
            calls["init"] = kwargs
            return "ready", True

    class FakeLLMHandler:
        def initialize(self, **kwargs: Any) -> tuple[str, bool]:
            calls["lm_init"] = kwargs
            return "lm ready", True

    def fake_generate_music(
        dit_handler: Any,
        llm_handler: Any,
        *,
        params: _FakeGenerationParams,
        config: _FakeGenerationConfig,
        save_dir: str | None = None,
    ) -> SimpleNamespace:
        calls["generate"] = {
            "dit_handler": dit_handler,
            "llm_handler": llm_handler,
            "params": params,
            "config": config,
            "save_dir": save_dir,
        }
        return SimpleNamespace(
            success=True,
            error=None,
            status_message="",
            audios=[{"tensor": torch.zeros(2, 480), "sample_rate": 48000}],
        )

    acestep_module = ModuleType("acestep")
    acestep_module.__path__ = []
    handler_module = ModuleType("acestep.handler")
    handler_module.AceStepHandler = FakeAceStepHandler
    llm_module = ModuleType("acestep.llm_inference")
    llm_module.LLMHandler = FakeLLMHandler
    inference_module = ModuleType("acestep.inference")
    inference_module.GenerationParams = _FakeGenerationParams
    inference_module.GenerationConfig = _FakeGenerationConfig
    inference_module.generate_music = fake_generate_music

    monkeypatch.setitem(sys.modules, "acestep", acestep_module)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_module)
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", llm_module)
    monkeypatch.setitem(sys.modules, "acestep.inference", inference_module)

    from anvil_audio.pipelines.acestep import ACEStepPipeline

    pipe = ACEStepPipeline(
        project_root=str(tmp_path),
        config_path="acestep-v15-sft",
        device="mps",
        lm_model_path="acestep-5Hz-lm-4B",
        default_params={
            "steps": 50,
            "cfg_scale": 7.5,
            "sampler_type": "ode",
            "shift": 3.0,
            "thinking": False,
            "use_cot_metas": False,
            "use_cot_caption": False,
            "use_cot_language": False,
            "dcw_enabled": False,
            "velocity_norm_threshold": 0.0,
            "velocity_ema_factor": 0.0,
        },
    )
    audio = pipe.generate(
        [
            {
                "prompt": "gritty alternative rock",
                "lyrics": "[Instrumental]",
                "seconds_total": 5,
                "negative_prompt": "muddy low fidelity",
            }
        ],
        seed=123,
    )

    assert audio.shape == (1, 2, 480)
    assert calls["init"]["config_path"] == "acestep-v15-sft"
    assert calls["init"]["use_mlx_dit"] is True
    assert calls["generate"]["llm_handler"] is pipe._lm_handler
    assert calls["generate"]["save_dir"] is None

    params = calls["generate"]["params"]
    assert params.caption == "gritty alternative rock"
    assert params.lyrics == "[Instrumental]"
    assert params.duration == 5.0
    assert params.inference_steps == 50
    assert params.guidance_scale == 7.5
    assert params.infer_method == "ode"
    assert params.shift == 3.0
    assert params.thinking is False
    assert params.use_cot_metas is False
    assert params.use_cot_caption is False
    assert params.use_cot_language is False
    assert params.lm_negative_prompt == "muddy low fidelity"
    assert not hasattr(params, "sampler_mode")
    assert not hasattr(params, "dcw_enabled")
    assert not hasattr(params, "velocity_norm_threshold")

    config = calls["generate"]["config"]
    assert config.batch_size == 1
    assert config.use_random_seed is False
    assert config.seeds == [123]
    assert config.audio_format == "wav"
