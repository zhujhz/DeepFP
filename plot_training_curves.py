"""Plot stored training histories; does not run training or evaluation."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def plot_loss(rows, dest, scenario):
    """Show the two loss definitions separately on elapsed-time axes."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    stages = (
        ("supervised", "Supervised: complex MSE", "#1565c0"),
        ("unsupervised", "Unsupervised: negative weighted rate", "#d45a00"),
    )
    for axis, (stage, title, color) in zip(axes, stages):
        subset = [row for row in rows if row["stage"] == stage]
        times = [float(row["elapsed_seconds"]) / 60 for row in subset]
        losses = [float(row["train_loss"]) for row in subset]
        axis.plot(times, losses, color=color, marker="o", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Elapsed training time (minutes)")
        axis.set_ylabel("Training loss")
        axis.grid(alpha=0.25)
    fig.suptitle(f"DeepFP {scenario.replace('_', ' ')} training loss")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(dest / f"train_loss_curve.{extension}", dpi=180)
    plt.close(fig)


def plot_one(root, scenario):
    history_path = root / "models" / scenario / "training_history.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No epochs in {history_path}")
    times_minutes = [float(row["elapsed_seconds"]) / 60 for row in rows]
    validation = [float(row["validation_wsr"]) for row in rows]
    rates = [float(row["lr"]) for row in rows]
    supervised_count = sum(row["stage"] == "supervised" for row in rows)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(times_minutes, validation, color="#1565c0", marker="o",
                 linewidth=2, label="Validation WSR")
    axes[0].set_ylabel("Validation weighted sum rate")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(times_minutes, rates, color="#d45a00", marker="o", linewidth=2)
    axes[1].set_ylabel("Learning rate")
    axes[1].set_xlabel("Elapsed training time (minutes)")
    axes[1].grid(alpha=0.25)
    if supervised_count and supervised_count < len(rows):
        boundary = times_minutes[supervised_count - 1]
        for axis in axes:
            axis.axvline(boundary, color="#777777", linestyle="--", linewidth=1,
                         label="End of supervision" if axis is axes[0] else None)
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle(f"DeepFP {scenario.replace('_', ' ')} training")
    fig.tight_layout()
    dest = root / "results" / scenario
    dest.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(dest / f"training_curve.{extension}", dpi=180)
    plt.close(fig)
    plot_loss(rows, dest, scenario)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scenario", choices=("all", "single_cell", "three_cell"),
                        default="all")
    args = parser.parse_args()
    scenarios = ("single_cell", "three_cell") if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        plot_one(args.root, scenario)
        print(args.root / "results" / scenario / "training_curve.png")


if __name__ == "__main__":
    main()
