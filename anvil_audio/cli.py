"""
Anvil Audio — unified CLI entry point.

Usage::

    anvil generate --model stable-audio-open-1.0 --prompt "wooden door creak"
    anvil generate --model sfx-v1 --cond-yaml-path batch.yaml --output-dir ./out
    anvil generate --list-models
    anvil generate --model-config path/to/config.json --ckpt-path path/to/ckpt.pt \\
        --prompt "rain on tin roof" --output-dir ./out

All subcommand logic lives in the dedicated modules; this file only dispatches.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        sys.exit(0)

    sub = sys.argv[1]
    # Remove the subcommand token so the delegated parser sees a clean argv.
    sys.argv = [f"anvil {sub}"] + sys.argv[2:]

    if sub == "generate":
        from anvil_audio._cli_generate import main as gen_main
        gen_main()
    else:
        print(f"anvil: unknown subcommand '{sub}'")
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    print(
        "Anvil Audio — pluggable AI audio generation\n"
        "\n"
        "Usage:  anvil <subcommand> [options]\n"
        "\n"
        "Subcommands:\n"
        "  generate    Generate audio from a model and prompt\n"
        "\n"
        "Run 'anvil <subcommand> --help' for subcommand options.\n"
    )


if __name__ == "__main__":
    main()
