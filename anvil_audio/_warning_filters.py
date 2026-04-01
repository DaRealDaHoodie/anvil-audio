"""
Warning filters for suppressing non-actionable upstream deprecation noise.

Imported and applied early in ``run_gradio.py`` and ``anvil_audio/cli.py``
**before** any heavy ML packages are loaded, so the filters are in place
when those packages run their module-level code.

Only ``Warning`` subclass instances are affected — all exceptions, errors,
and our own ``print`` output are unaffected.

Pass ``--verbose`` on the command line to disable these filters entirely.
"""

from __future__ import annotations

import warnings


# Packages whose FutureWarning noise we suppress.
# The module pattern is matched with re.match (start-anchored), so "torch"
# covers torch, torch.*, torchsde, torchaudio, and any other torch* package.
_FUTURE_WARNING_MODULES: tuple[str, ...] = (
    r"torch",           # torch + torchsde + torchaudio (all start with "torch")
    r"x_transformers",
    r"clip",
    r"pyloudnorm",
)

# Packages whose DeprecationWarning we suppress.
# pkg_resources is deprecated in Python 3.12+ and emits a warning when any
# third-party package (e.g. clip) imports it.
_DEPRECATION_MODULES: tuple[str, ...] = (
    r"pkg_resources",
    r"clip",
)

# UserWarning message patterns — matched against the warning message text.
_USER_WARNING_PATTERNS: tuple[str, ...] = (
    r".*flash.?attn.*",          # "flash_attn not installed" from x_transformers / others
    r".*xformers.*not installed.*",
)


def apply_filters() -> None:
    """Install warning filters for known-noisy upstream packages.

    Safe to call multiple times — ``warnings.filterwarnings`` is idempotent
    when called with the same arguments (it deduplicates the filter list).
    """
    for module in _FUTURE_WARNING_MODULES:
        warnings.filterwarnings("ignore", category=FutureWarning, module=module)

    for module in _DEPRECATION_MODULES:
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=module)

    for pattern in _USER_WARNING_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern, category=UserWarning)
