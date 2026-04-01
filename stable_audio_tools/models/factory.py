from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stable_audio_tools.core.pipeline import DiffusionPipeline


def create_pipeline_from_config(
    model_config: dict[str, Any],
    ckpt_path: str | None = None,
    pretransform_ckpt_path: str | None = None,
    default_params: dict[str, Any] | None = None,
) -> "DiffusionPipeline":
    """Translate an existing JSON model config into a ``DiffusionPipeline``.

    This is the bridge between the legacy config-based loading path and the
    new component/pipeline system.  The old ``create_model_from_config`` path
    is used internally; callers get back a clean ``BasePipeline`` interface.

    All existing scripts that build models via ``create_model_from_config``
    continue to work unchanged — this function is additive only.

    Args:
        model_config:            Parsed JSON model config dict.
        ckpt_path:               Optional path to a local checkpoint to load
                                 weights from after model creation.
        pretransform_ckpt_path:  Optional separate checkpoint for the
                                 pretransform (VAE stage).
        default_params:          Generation parameter overrides for this
                                 pipeline (steps, cfg_scale, sampler_type,
                                 sigma_min, sigma_max).

    Returns:
        A ``DiffusionPipeline`` wrapping the constructed model.

    Raises:
        NotImplementedError: If the model type is not ``diffusion_cond`` or
                             a compatible variant.  Use ``create_model_from_config``
                             directly for autoencoders, LMs, and other types.
    """
    from stable_audio_tools.core.pipeline import DiffusionPipeline
    from stable_audio_tools.models.utils import load_ckpt_state_dict
    from stable_audio_tools.utils.torch_common import copy_state_dict

    supported = {"diffusion_cond", "diffusion_cond_inpaint", "diffusion_prior"}
    model_type = model_config.get("model_type", "")
    if model_type not in supported:
        raise NotImplementedError(
            f"create_pipeline_from_config only supports {sorted(supported)}; "
            f"got '{model_type}'.  Use create_model_from_config for other types."
        )

    model = create_model_from_config(model_config)

    if ckpt_path is not None:
        copy_state_dict(model, load_ckpt_state_dict(ckpt_path))

    if pretransform_ckpt_path is not None:
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(pretransform_ckpt_path), strict=False
        )

    return DiffusionPipeline(
        model=model,
        model_config=model_config,
        default_params=default_params,
    )


def create_pipeline_from_config_path(
    model_config_path: str,
    ckpt_path: str | None = None,
    pretransform_ckpt_path: str | None = None,
    default_params: dict[str, Any] | None = None,
) -> "DiffusionPipeline":
    """Convenience wrapper: load a JSON file then call ``create_pipeline_from_config``.

    Args:
        model_config_path:      Path to a JSON model config file.
        ckpt_path:              See ``create_pipeline_from_config``.
        pretransform_ckpt_path: See ``create_pipeline_from_config``.
        default_params:         See ``create_pipeline_from_config``.

    Returns:
        A ``DiffusionPipeline``.
    """
    with open(model_config_path) as fh:
        model_config = json.load(fh)
    return create_pipeline_from_config(
        model_config,
        ckpt_path=ckpt_path,
        pretransform_ckpt_path=pretransform_ckpt_path,
        default_params=default_params,
    )


def create_model_from_config(model_config):
    model_type = model_config['model_type']

    if model_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config)
    elif model_type == 'diffusion_uncond':
        from .diffusion import create_diffusion_uncond_from_config
        return create_diffusion_uncond_from_config(model_config)
    elif model_type == 'diffusion_cond' or model_type == 'diffusion_cond_inpaint' or model_type == "diffusion_prior":
        from .diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config)
    elif model_type == 'diffusion_autoencoder':
        from .autoencoders import create_diffAE_from_config
        return create_diffAE_from_config(model_config)
    elif model_type == 'lm':
        from .lm import create_audio_lm_from_config
        return create_audio_lm_from_config(model_config)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')


def create_model_from_config_path(model_config_path):
    with open(model_config_path) as f:
        model_config = json.load(f)

    return create_model_from_config(model_config)


def create_pretransform_from_config(pretransform_config, sample_rate):
    pretransform_type = pretransform_config['type']

    if pretransform_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        from .pretransforms import AutoencoderPretransform

        # Create fake top-level config to pass sample rate to autoencoder constructor
        # This is a bit of a hack but it keeps us from re-defining the sample rate in the config
        autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_config["config"]}
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)
        iterate_batch = pretransform_config.get("iterate_batch", False)
        chunked = pretransform_config.get("chunked", False)

        pretransform = AutoencoderPretransform(autoencoder, scale=scale, model_half=model_half, iterate_batch=iterate_batch, chunked=chunked)
    elif pretransform_type == 'wavelet':
        from .pretransforms import WaveletPretransform

        wavelet_config = pretransform_config["config"]
        channels = wavelet_config["channels"]
        levels = wavelet_config["levels"]
        wavelet = wavelet_config["wavelet"]

        pretransform = WaveletPretransform(channels, levels, wavelet)
    elif pretransform_type == 'pqmf':
        from .pretransforms import PQMFPretransform
        pqmf_config = pretransform_config["config"]
        pretransform = PQMFPretransform(**pqmf_config)
    elif pretransform_type == 'dac_pretrained':
        from .pretransforms import PretrainedDACPretransform
        pretrained_dac_config = pretransform_config["config"]
        pretransform = PretrainedDACPretransform(**pretrained_dac_config)
    elif pretransform_type == "audiocraft_pretrained":
        from .pretransforms import AudiocraftCompressionPretransform

        audiocraft_config = pretransform_config["config"]
        pretransform = AudiocraftCompressionPretransform(**audiocraft_config)
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')

    enable_grad = pretransform_config.get('enable_grad', False)
    pretransform.enable_grad = enable_grad

    pretransform.eval().requires_grad_(pretransform.enable_grad)

    return pretransform


def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config['type']

    if bottleneck_type == 'tanh':
        from .bottleneck import TanhBottleneck
        bottleneck = TanhBottleneck()
    elif bottleneck_type == 'vae':
        from .bottleneck import VAEBottleneck
        bottleneck = VAEBottleneck()
    elif bottleneck_type == 'rvq':
        from .bottleneck import RVQBottleneck
        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }
        quantizer_params.update(bottleneck_config["config"])
        bottleneck = RVQBottleneck(**quantizer_params)
    elif bottleneck_type == "dac_rvq":
        from .bottleneck import DACRVQBottleneck
        bottleneck = DACRVQBottleneck(**bottleneck_config["config"])
    elif bottleneck_type == 'rvq_vae':
        from .bottleneck import RVQVAEBottleneck
        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }
        quantizer_params.update(bottleneck_config["config"])
        bottleneck = RVQVAEBottleneck(**quantizer_params)
    elif bottleneck_type == 'dac_rvq_vae':
        from .bottleneck import DACRVQVAEBottleneck
        bottleneck = DACRVQVAEBottleneck(**bottleneck_config["config"])
    elif bottleneck_type == 'l2_norm':
        from .bottleneck import L2Bottleneck
        bottleneck = L2Bottleneck()
    elif bottleneck_type == "wasserstein":
        from .bottleneck import WassersteinBottleneck
        bottleneck = WassersteinBottleneck(**bottleneck_config.get("config", {}))
    elif bottleneck_type == "fsq":
        from .bottleneck import FSQBottleneck
        bottleneck = FSQBottleneck(**bottleneck_config["config"])
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')

    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck
