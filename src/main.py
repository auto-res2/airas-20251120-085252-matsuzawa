"""src/main.py – Hydra orchestrator that spawns src.train as subprocess."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("Launching training subprocess:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
