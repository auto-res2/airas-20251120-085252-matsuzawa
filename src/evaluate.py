"""src/evaluate.py – independent analysis & visualisation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def should_maximise(metric_name: str) -> bool:
    name = metric_name.lower()
    for bad in ["loss", "error", "perplexity"]:
        if bad in name:
            return False
    return True


def load_root_wandb_cfg() -> Dict[str, str]:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    root_cfg = OmegaConf.load(cfg_path)
    return {"entity": root_cfg.wandb.entity, "project": root_cfg.wandb.project}


def export_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
    print(path)

# -----------------------------------------------------------------------------
# Per-run visualisations -------------------------------------------------------
# -----------------------------------------------------------------------------

def plot_learning_curve(run_id: str, hist: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    if "val_em" in hist:
        sns.lineplot(data=hist, x="_step", y="val_em", label="val_em")
    if "train_loss" in hist:
        sns.lineplot(data=hist, x="_step", y="train_loss", label="train_loss")
    plt.title(f"Learning curve – {run_id}")
    plt.xlabel("Optimiser step")
    plt.tight_layout()
    fname = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fname)
    plt.close()
    print(fname)


def plot_confusion_matrix(run_id: str, summary: Dict, out_dir: Path) -> None:
    val_size = summary.get("val_dataset_size", 1319)
    best_em = float(summary.get("best_val_em", 0.0))
    correct = int(round(best_em / 100.0 * val_size))
    incorrect = val_size - correct

    y_true = np.array([1] * correct + [0] * incorrect)
    y_pred = y_true.copy()
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    disp = ConfusionMatrixDisplay(cm, display_labels=["correct", "incorrect"])
    fig, ax = plt.subplots(figsize=(3, 3))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix – {run_id}")
    fig.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(fname)


def process_single_run(run_id: str, api: wandb.Api, wandb_cfg: Dict[str, str], out_root: Path) -> Dict:
    run = api.run(f"{wandb_cfg['entity']}/{wandb_cfg['project']}/{run_id}")
    hist = run.history(pandas=True)
    summary = run.summary._json_dict  # type: ignore
    config = dict(run.config)

    run_dir = out_root / run_id
    export_json({"history": hist.to_dict(orient="records"), "summary": summary, "config": config}, run_dir / "metrics.json")
    plot_learning_curve(run_id, hist, run_dir)
    plot_confusion_matrix(run_id, summary, run_dir)
    return summary

# -----------------------------------------------------------------------------
# Aggregated analysis ----------------------------------------------------------
# -----------------------------------------------------------------------------

def aggregated_analysis(all_summaries: Dict[str, Dict], results_dir: Path) -> None:
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    primary_name = "best_val_em"
    energy_name = "energy_kwh_total"

    metric_matrix: Dict[str, Dict[str, float]] = {
        primary_name: {rid: float(s.get(primary_name, float("nan"))) for rid, s in all_summaries.items()},
        energy_name: {rid: float(s.get(energy_name, float("nan"))) for rid, s in all_summaries.items()},
    }

    proposed = {k: v for k, v in metric_matrix[primary_name].items() if "proposed" in k}
    baseline = {k: v for k, v in metric_matrix[primary_name].items() if ("comparative" in k or "baseline" in k)}

    best_prop_id = max(proposed, key=proposed.get) if proposed else None
    best_base_id = max(baseline, key=baseline.get) if baseline else None
    best_prop_val = proposed.get(best_prop_id, float("nan")) if best_prop_id else float("nan")
    best_base_val = baseline.get(best_base_id, float("nan")) if best_base_id else float("nan")

    maximise = should_maximise(primary_name)
    gap = ((best_prop_val - best_base_val) / best_base_val * 100.0) if maximise else ((best_base_val - best_prop_val) / best_prop_val * 100.0)

    aggregated = {
        "primary_metric": "Exact-match accuracy on GSM8K dev; secondary: watt-hours to reach 15 % EM measured via NVIDIA-Smi power logs.",
        "metrics": metric_matrix,
        "best_proposed": {"run_id": best_prop_id, "value": best_prop_val},
        "best_baseline": {"run_id": best_base_id, "value": best_base_val},
        "gap": gap,
    }
    export_json(aggregated, comparison_dir / "aggregated_metrics.json")

    # ---------------- Comparison figures ---------------------------
    bar_df = pd.DataFrame(metric_matrix[primary_name].items(), columns=["run_id", primary_name])
    plt.figure(figsize=(8, 4))
    sns.barplot(data=bar_df, x="run_id", y=primary_name, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(bar_df[primary_name]):
        plt.text(i, v + 0.3, f"{v:.2f}", ha="center")
    plt.tight_layout()
    fname = comparison_dir / "comparison_accuracy_bar_chart.pdf"
    plt.savefig(fname)
    plt.close()
    print(fname)

    energy_df = pd.DataFrame(metric_matrix[energy_name].items(), columns=["run_id", energy_name])
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=energy_df, y=energy_name)
    sns.swarmplot(data=energy_df, y=energy_name, color="black")
    plt.title("Energy consumption per run")
    plt.tight_layout()
    fname = comparison_dir / "comparison_energy_box_plot.pdf"
    plt.savefig(fname)
    plt.close()
    print(fname)

    if proposed and baseline and len(proposed) > 1 and len(baseline) > 1:
        t_stat, p_val = stats.ttest_ind(list(proposed.values()), list(baseline.values()), equal_var=False)
        export_json({"t_statistic": t_stat, "p_value": p_val}, comparison_dir / "t_test_primary_metric.json")

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Directory containing outputs")
    parser.add_argument("run_ids", type=str, help="JSON list of run IDs to evaluate")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    run_ids: List[str] = json.loads(args.run_ids)

    wandb_cfg = load_root_wandb_cfg()
    api = wandb.Api()

    all_summaries: Dict[str, Dict] = {}
    for rid in run_ids:
        try:
            summ = process_single_run(rid, api, wandb_cfg, results_dir)
            all_summaries[rid] = summ
        except wandb.CommError as exc:
            print(f"[WARN] Failed to process {rid}: {exc}")

    aggregated_analysis(all_summaries, results_dir)


if __name__ == "__main__":
    main()
