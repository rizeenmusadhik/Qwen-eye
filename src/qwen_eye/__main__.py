"""python -m qwen_eye ...: the `qwen-eye` command (cli.py).

`python -m qwen_eye run <run_dir>` runs <run_dir>/plan.json, e.g. inside a cluster job; its
"kind" says which job: "attn" (maps A/B, the default), "gradcam" (maps E, H, D, C, W) or
"deletion" (deletion test of saved maps)."""

from __future__ import annotations

import sys

from qwen_eye.cli import main

if __name__ == "__main__":
    sys.exit(main())
