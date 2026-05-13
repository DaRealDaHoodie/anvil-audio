from anvil_audio.core.registry import ModelRegistry


def test_acestep_sft_defaults_match_known_good_direct_path(monkeypatch, tmp_path):
    monkeypatch.setenv("ACESTEP_PROJECT_ROOT", str(tmp_path))

    entry = ModelRegistry().get("acestep-v1.5-sft")
    params = entry.resolved_params()

    assert entry.model_config_path == "acestep-v15-sft"
    assert entry.lm_model_path == "acestep-5Hz-lm-4B"
    assert params["steps"] == 50
    assert params["cfg_scale"] == 7.5
    assert params["sampler_type"] == "ode"
    assert params["shift"] == 3.0
    assert params["thinking"] is False
    assert params["use_cot_metas"] is False
    assert params["use_cot_caption"] is False
    assert params["use_cot_language"] is False
    assert params["dcw_enabled"] is False
    assert params["velocity_norm_threshold"] == 0.0
    assert params["velocity_ema_factor"] == 0.0
