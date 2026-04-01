"""
Optional pipeline adapters for third-party generative models.

Each adapter wraps an external model as a ``BasePipeline`` so it can be
loaded via the Anvil registry, driven by the CLI, and rendered in the
Gradio UI exactly like the built-in Stable Audio pipelines.

Available adapters
------------------
acestep  — ACE-Step v1.5 music generation model
          ``from anvil_audio.pipelines.acestep import ACEStepPipeline``
"""
