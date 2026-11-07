"""Compare three-cell DeepFP, FastFP, and the existing multicell WMMSE."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from FP_unfolding.FastFP_net import FastFPnet
from FP_unfolding.FastFP_alg import FastFP
from FP_unfolding.WMMSE import WMMSE
from FP_unfolding.compute_WSR import WSR2
from train_threecell import load_split


ROOT = Path(__file__).resolve().parent
WEIGHTS = [i / 21 for i in range(1, 7)]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data" / "three_cell")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models" / "three_cell" / "deepfp_hybrid_best.pth")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "three_cell" / "comparison.json")
    parser.add_argument("--limit", type=int, default=50,
                        help="Test samples (default 50 for a practical WMMSE demo; 0 uses all 5001)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--fastfp-iters", type=int, default=30)
    parser.add_argument("--wmmse-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run train_threecell.py first.")
    device = torch.device("cuda:0" if (args.device == "auto" and torch.cuda.is_available())
                          or args.device == "cuda" else "cpu")
    channels = load_split(args.data_dir, "test", args.limit)
    count = len(channels)
    model = FastFPnet(3, 6, 4, 64, WEIGHTS, 2, 8, device, hidden_size=64,
                      signal_max_power=100, noise_power=1e-8).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    fastfp = FastFP().to(device)
    wmmse = WMMSE().to(device)
    methods = ("DeepFP", "FastFP", "WMMSE")
    scores = {name: [] for name in methods}
    powers = {name: [] for name in methods}
    seconds = {name: 0.0 for name in methods}
    rows = []
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            h = channels[offset:min(offset+args.batch_size, count)].to(device)
            torch.manual_seed(args.seed + 2000000 + offset)
            init_v = model.init_input(len(h)).to(device)
            for name in methods:
                sync(device)
                started = time.perf_counter()
                if name == "DeepFP":
                    v, _ = model(h, action_noise_power=0, init_V=init_v)
                elif name == "FastFP":
                    v = fastfp.forward(h, WEIGHTS, args.fastfp_iters, d=2,
                                       signal_max_power=100, noise_power=1e-8,
                                       init_V=init_v)
                else:
                    # WMMSE.forward is the single-cell variant. The existing
                    # multicell update is single_iteration. Its implementation
                    # indexes an internal matrix by sample, so use batch size 1.
                    solved = []
                    for j in range(len(h)):
                        current = init_v[j:j+1].clone()
                        for _ in range(args.wmmse_iters):
                            current = wmmse.single_iteration(
                                h[j:j+1], current, WEIGHTS,
                                signal_max_power=100, noise_power=1e-8)
                        solved.append(current)
                    v = torch.cat(solved, dim=0)
                    # The existing bisection stops at 1e-4 power tolerance.
                    # Remove only a possible rounding/tolerance overshoot.
                    per_cell_power = v.abs().square().sum(dim=(-1, -2, -3))
                    scale = (100 / per_cell_power).sqrt().clamp(max=1)
                    v = v * scale[:, :, None, None, None]
                sync(device)
                seconds[name] += time.perf_counter() - started
                batch_scores = WSR2(h, v, WEIGHTS, noise_power=1e-8).tolist()
                cell_power = v.abs().square().sum(dim=(-1, -2, -3))
                batch_power = cell_power.max(dim=-1).values.tolist()
                scores[name].extend(batch_scores)
                powers[name].extend(batch_power)
                rows.extend({"index": offset+j, "method": name,
                             "wsr": batch_scores[j], "max_cell_power": batch_power[j]}
                            for j in range(len(h)))
            print("evaluated", offset + len(h), "/", count, flush=True)
    summary = {name: {"mean_wsr": float(np.mean(scores[name])),
                      "std_wsr": float(np.std(scores[name])),
                      "mean_max_cell_power": float(np.mean(powers[name])),
                      "max_cell_power": float(np.max(powers[name])),
                      "algorithm_seconds": seconds[name]} for name in methods}
    result = {"data_dir": str(args.data_dir), "checkpoint": str(args.checkpoint),
              "samples": count, "device": str(device), "batch_size": args.batch_size,
              "fastfp_iterations": args.fastfp_iters,
              "wmmse_iterations": args.wmmse_iters,
              "wmmse_entry_point": "WMMSE.single_iteration (per sample)",
              "seed": args.seed, "weights": WEIGHTS, "power": 100,
              "noise": 1e-8, "score_scale": 1, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "method", "wsr", "max_cell_power"])
        writer.writeheader()
        writer.writerows(rows)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
