"""src/train.py – comprehensive training with Hydra, WandB & µ-PACT
This revision fixes the two data-pipeline bugs reported by the validator:
1. The GSM8K split string coming from run-configs (e.g. "main/train") is now
   parsed correctly so that HuggingFace receives subset="main", split="train".
2. The question / answer columns are preserved during tokenisation so that
   greedy evaluation can access them at validation time.
"""
from __future__ import annotations

import copy
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import optuna
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Make project-root importable (Hydra changes CWD) -----------------------------
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import load_model_and_tokenizer  # noqa: E402
from src.preprocess import GSM8KPreprocessor, build_collate_fn  # noqa: E402

# -----------------------------------------------------------------------------
# Reproducibility --------------------------------------------------------------
# -----------------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

# -----------------------------------------------------------------------------
# GPU power helper -------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_gpu_power_watts() -> float:
    """Return instantaneous aggregate GPU power draw in watts; 0 if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode()
        return sum(float(v) for v in out.strip().split("\n") if v.strip())
    except Exception:  # pylint: disable=broad-except
        return 0.0

# -----------------------------------------------------------------------------
# µ-PACT optimiser -------------------------------------------------------------
# -----------------------------------------------------------------------------

class MuPACT:
    """Block-wise PAC optimiser with auto step-size (see paper)."""

    def __init__(
        self,
        optim: torch.optim.Optimizer,
        model: torch.nn.Module,
        *,
        blocks: int,
        delta: float = 0.05,
        rho: float = 0.9,
        ga_max: int = 8,
        eta_max: float = 5e-5,
        eps: float = 1e-8,
    ) -> None:
        import scipy.stats  # heavy import: done lazily

        self.optim = optim
        self.model = model
        self.B = int(blocks)
        self.delta = float(delta)
        self.rho = float(rho)
        self.ga_max = int(ga_max)
        self.eta_max = float(eta_max)
        self.eps = float(eps)

        self.q_delta = scipy.stats.chi2.isf(self.delta, df=1) / (1 - self.rho**2)
        self.params: List[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        assert self.B <= len(self.params), "µ-PACT: B must not exceed number of trainable tensors"

        self.prev_step_norm = 1.0
        self._reset()

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self.g_sum: List[torch.Tensor | None] = [None] * self.B
        self.g2_sum: List[float] = [0.0] * self.B
        self.micro_steps: int = 0

    # ------------------------------------------------------------------
    def _flat_grad_blocks(self) -> List[torch.Tensor]:
        flat = torch.cat([p.grad.view(-1) for p in self.params if p.grad is not None])
        n_total = flat.numel()
        blk_size = math.ceil(n_total / self.B)
        blocks: List[torch.Tensor] = [
            flat[i * blk_size : min(n_total, (i + 1) * blk_size)] for i in range(self.B)
        ]
        if blocks[-1].numel() == 0:
            blocks[-1] = torch.zeros(1, device=flat.device)
        return blocks

    # ------------------------------------------------------------------
    def end_micro(self) -> bool:
        """Call after each backward().  Returns True when an optimiser step fires."""
        self.micro_steps += 1
        all_pass = True
        for b_idx, g_b in enumerate(self._flat_grad_blocks()):
            g_norm = torch.norm(g_b)
            if self.g_sum[b_idx] is None:
                self.g_sum[b_idx] = g_b.detach().clone()
            else:
                self.g_sum[b_idx].add_(g_b)
            self.g2_sum[b_idx] += float(g_norm**2)

            mu_b = self.g_sum[b_idx] / self.micro_steps  # type: ignore[arg-type]
            mu_norm = torch.norm(mu_b)
            var_b = (
                self.g2_sum[b_idx] - self.micro_steps * float(mu_norm**2)
            ) / max(1, self.micro_steps - 1)
            T_b = (mu_norm**2) / (var_b / self.micro_steps + self.eps)
            all_pass &= bool(T_b >= self.q_delta)

        if all_pass or self.micro_steps >= self.ga_max:
            # ------------------- Gradient integrity assertions -----------------
            grads = [p.grad for p in self.params]
            assert all(g is not None for g in grads), "µ-PACT: some gradients vanished"
            total_norm = torch.norm(torch.stack([g.detach().norm() for g in grads]))  # type: ignore[arg-type]
            assert total_norm.item() > 0, "µ-PACT: zero gradients before optimiser.step()"

            # ------------------- Lipschitz-based learning rate ------------------
            g_tot = torch.sqrt(sum(gs.norm() ** 2 for gs in self.g_sum if gs is not None))  # type: ignore[arg-type]
            L_hat = g_tot / (self.prev_step_norm + self.eps)
            alpha = math.acos(self.rho)
            lr = min(self.eta_max, 2 * math.cos(alpha) / (float(L_hat) + self.eps))
            for pg in self.optim.param_groups:
                pg["lr"] = lr

            # ------------------- Parameter update ------------------------------
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
            self.prev_step_norm = float(
                torch.sqrt(sum(p.data.norm() ** 2 for p in self.params))  # type: ignore[arg-type]
            ) + self.eps
            self._reset()
            return True
        return False

# -----------------------------------------------------------------------------
# Greedy GSM8K evaluation ------------------------------------------------------
# -----------------------------------------------------------------------------
_NUMERIC_RE = re.compile(r"-?\d+\.?\d*")


def _canonical_number(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()
    found = _NUMERIC_RE.findall(text)
    return found[-1] if found else text.strip()


def greedy_eval(model, tokenizer, ds, device, *, max_new_tokens: int = 256) -> float:
    """Greedy decode evaluation (exact-match %)."""
    model.eval()
    correct = 0
    for ex in ds:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        pred = tokenizer.decode(out[0][inp.input_ids.shape[1] :], skip_special_tokens=True)
        correct += int(_canonical_number(pred) == _canonical_number(ex["answer"]))
    model.train()
    return correct / len(ds) * 100.0

# -----------------------------------------------------------------------------
# Core experiment --------------------------------------------------------------
# -----------------------------------------------------------------------------

def run_single_experiment(cfg: DictConfig, *, enable_wandb: bool = True) -> Dict[str, Any]:
    set_random_seed(42)

    # ---------------- WandB initialisation -------------------------
    if enable_wandb and cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=str(cfg.run_id),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print("WandB URL:", wandb_run.url)
    else:
        wandb_run = None  # type: ignore

    # ---------------- Model & tokenizer ----------------------------
    tokenizer, model = load_model_and_tokenizer(cfg)
    assert tokenizer.pad_token_id is not None, "pad_token_id must be set after loading"

    # ---------------- Dataset loading ------------------------------
    preproc = GSM8KPreprocessor(tokenizer, cfg)
    train_ds = preproc.get_split(cfg.dataset.split.train)
    val_ds = preproc.get_split(cfg.dataset.split.validation)

    if cfg.mode == "trial":
        # tight subset for CI / quick validation
        train_ds = train_ds.select(range(min(64, len(train_ds))))
        val_ds = val_ds.select(range(min(64, len(val_ds))))
        cfg.training.physical_batch_size = 2

    collate_fn = build_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.physical_batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------------- Optimiser selection --------------------------
    base_lr = float(getattr(cfg.training, "learning_rate_base", 5e-5))
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=tuple(cfg.training.optimizer.betas),
        weight_decay=float(cfg.training.optimizer.weight_decay),
    )

    use_mupact = str(cfg.method).lower().startswith("proposed")
    if use_mupact:
        mupact = MuPACT(
            optim,
            model,
            blocks=int(cfg.training.mupact.B),
            delta=float(cfg.training.mupact.delta),
            rho=float(cfg.training.mupact.rho),
            ga_max=int(cfg.training.mupact.ga_max),
            eta_max=float(cfg.training.mupact.eta_max),
            eps=float(cfg.training.mupact.eps),
        )
    else:
        grad_accum_steps = int(cfg.training.gradient_accumulation_steps)
        accum_counter = 0

    global_step = 0
    best_val_em = 0.0
    energy_kwh = 0.0
    last_time = time.time()

    for epoch in range(int(cfg.training.epochs)):
        for step, batch in enumerate(train_loader):
            if cfg.mode == "trial" and step > 2:
                break

            if epoch == 0 and step == 0:
                # -------- Critical batch-start assertions --------
                assert batch["input_ids"].shape == batch["labels"].shape, "input/label shape mismatch"

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
            loss = model(**batch).loss
            if not use_mupact:
                loss = loss / grad_accum_steps
            loss.backward()

            if use_mupact:
                stepped = mupact.end_micro()
            else:
                accum_counter += 1
                stepped = False
                if accum_counter % grad_accum_steps == 0:
                    # -------- Pre-optimiser gradient assertions ----
                    grads = [p.grad for p in model.parameters() if p.requires_grad]
                    assert all(g is not None for g in grads), "Gradients became None"
                    total_norm = torch.norm(torch.stack([g.detach().norm() for g in grads]))  # type: ignore[arg-type]
                    assert total_norm.item() > 0, "Zero gradients before step()"

                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    accum_counter = 0
                    stepped = True

            # ---------------- Energy accounting -------------------
            now = time.time()
            energy_kwh += get_gpu_power_watts() * (now - last_time) / 3_600_000
            last_time = now

            # ---------------- WandB per-batch log ------------------
            if wandb_run is not None:
                wandb.log({"train_loss": float(loss)}, step=global_step)

            # ---------------- Validation --------------------------
            if stepped:
                global_step += 1
                if (
                    global_step % int(cfg.evaluation.eval_every_updates) == 0
                    or (cfg.mode == "trial" and global_step == 1)
                ):
                    subset = val_ds.select(range(min(128, len(val_ds))))
                    val_em = greedy_eval(model, tokenizer, subset, device)
                    best_val_em = max(best_val_em, val_em)
                    if wandb_run is not None:
                        wandb.log({"val_em": val_em, "energy_kwh": energy_kwh}, step=global_step)
        if cfg.mode == "trial":
            break

    if wandb_run is not None:
        wandb_run.summary["best_val_em"] = best_val_em
        wandb_run.summary["energy_kwh_total"] = energy_kwh
        wandb_run.finish()

    return {"best_val_em": best_val_em, "energy_kwh": energy_kwh}

# -----------------------------------------------------------------------------
# Optuna objective -------------------------------------------------------------
# -----------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    cfg = copy.deepcopy(base_cfg)

    def assign(cfg_dict: DictConfig, dotted: str, value: Any) -> None:
        cur = cfg_dict
        for p in dotted.split(".")[:-1]:
            cur = cur[p]
        cur[dotted.split(".")[-1]] = value

    for path, spec in cfg.optuna.search_space.items():
        if spec["type"] == "int":
            val = trial.suggest_int(path, spec["low"], spec["high"], step=spec.get("step", 1))
        elif spec["type"] == "uniform":
            val = trial.suggest_float(path, spec["low"], spec["high"])
        elif spec["type"] == "loguniform":
            val = trial.suggest_float(path, spec["low"], spec["high"], log=True)
        elif spec["type"] == "categorical":
            val = trial.suggest_categorical(path, spec["choices"])
        else:
            raise ValueError(f"Unknown Optuna type: {spec['type']}")
        assign(cfg, path, val)

    # lightweight objective
    cfg.training.epochs = 1
    cfg.evaluation.eval_every_updates = 50
    cfg.wandb.mode = "disabled"

    metrics = run_single_experiment(cfg, enable_wandb=False)
    return metrics["best_val_em"]  # maximise

# -----------------------------------------------------------------------------
# Hydra entry-point ------------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:  # noqa: D401
    root = Path(get_original_cwd())
    run_cfg_file = root / "config" / "runs" / f"{cfg.run}.yaml"
    assert run_cfg_file.exists(), f"Run-config not found: {run_cfg_file}"

    run_cfg = OmegaConf.load(run_cfg_file)
    cfg = OmegaConf.merge(cfg, run_cfg)

    # --------------- Mode-specific overrides ------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.training.epochs = 1
        cfg.evaluation.eval_every_updates = 10
        cfg.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    Path(cfg.results_dir).expanduser().mkdir(parents=True, exist_ok=True)

    # --------------- Optuna or direct run --------------------------
    if int(cfg.optuna.n_trials) > 0 and cfg.mode != "trial":
        study = optuna.create_study(direction=str(cfg.optuna.direction))
        study.optimize(lambda t: optuna_objective(t, cfg), n_trials=int(cfg.optuna.n_trials))
        print("Optuna best value:", study.best_value)
        for k, v in study.best_params.items():
            cur = cfg
            for p in k.split(".")[:-1]:
                cur = cur[p]
            cur[k.split(".")[-1]] = v
        run_single_experiment(cfg, enable_wandb=True)
    else:
        run_single_experiment(cfg, enable_wandb=True)


if __name__ == "__main__":
    main()
