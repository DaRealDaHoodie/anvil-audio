"""
Backward-compatible shim — delegates to anvil_audio._cli_generate.

Prefer the 'anvil generate' CLI entry point for new usage.
"""
from anvil_audio._cli_generate import main

if __name__ == "__main__":
    main()
