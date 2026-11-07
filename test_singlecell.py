"""Compare DeepFP, FastFP, and WMMSE-SC on the held-out single-cell split."""
import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from FP_unfolding.FastFP_net import FastFPnet
from FP_unfolding.FastFP_alg import FastFP
from FP_unfolding.WMMSE import WMMSE
from FP_unfolding.compute_WSR import WSR2


from train_singlecell import WEIGHTS


ROOT = Path(__file__).resolve().parent


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description="Single-cell DeepFP/FastFP/WMMSE comparison")
    parser.add_argument("--data", type=Path, default=ROOT / "Data" / "single_cell" / "test.h5")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models" / "single_cell" / "deepfp_hybrid_best.pth")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "single_cell" / "comparison.json")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of test channels (default 0 uses all)")
    parser.add_argument("--fastfp-iters", type=int, default=100)
    parser.add_argument("--wmmse-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    weights = WEIGHTS
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Train first or pass --checkpoint.")
    config_path = args.checkpoint.parent / "train_config.json"
    if config_path.exists():
        trained_weights = json.loads(config_path.read_text(encoding="utf-8"))["weights"]
        if not np.allclose(trained_weights, weights, rtol=0, atol=1e-12):
            parser.error("Checkpoint training weights do not match the original ascending weights")
    device = torch.device("cuda:0" if (args.device == "auto" and torch.cuda.is_available())
                          or args.device == "cuda" else "cpu")
    with h5py.File(args.data, "r") as f:
        size = min(len(f["H"]), args.limit) if args.limit else len(f["H"])
        if f["H"].shape[1:] != (1, 1, 6, 4, 64):
            parser.error(f"Unexpected H shape: {f['H'].shape}")
    model = FastFPnet(1, 6, 4, 64, weights, 2, 10, device, hidden_size=64,
                      signal_max_power=100, noise_power=1e-8).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    fastfp, wmmse = FastFP().to(device), WMMSE().to(device)
    methods = ("DeepFP", "FastFP", "WMMSE-SC")
    scores = {name: [] for name in methods}
    powers = {name: [] for name in methods}
    seconds = {name: 0.0 for name in methods}
    rows = []
    with h5py.File(args.data, "r") as f, torch.no_grad():
        for offset in range(0, size, args.batch_size):
            h = torch.from_numpy(np.array(f["H"][offset:min(offset+args.batch_size, size)], copy=True)).to(device)
            torch.manual_seed(args.seed + 2000000 + offset)
            v0 = model.init_input(len(h))
            for name in methods:
                sync(device)
                started = time.perf_counter()
                if name == "DeepFP":
                    v, _ = model(h, action_noise_power=0, init_V=v0)
                elif name == "FastFP":
                    v = fastfp.forward(h, weights, args.fastfp_iters, d=2,
                                       signal_max_power=100, noise_power=1e-8, init_V=v0)
                else:
                    v = wmmse.forward(h, weights, args.wmmse_iters, d=2,
                                      signal_max_power=100, noise_power=1e-8, init_V=v0)
                sync(device)
                seconds[name] += time.perf_counter() - started
                batch_scores = (WSR2(h, v, weights, noise_power=1e-8) * 6).tolist()
                batch_power = v.abs().square().sum(dim=(-1, -2, -3)).max(dim=-1).values.tolist()
                scores[name].extend(batch_scores)
                powers[name].extend(batch_power)
                rows.extend({"index": offset + j, "method": name, "wsr": batch_scores[j],
                             "cell_power": batch_power[j]} for j in range(len(h)))
            if offset == 0 or (offset + len(h)) % 500 == 0 or offset + len(h) == size:
                print("evaluated", offset + len(h), "/", size, flush=True)
    summary = {name: {"mean_wsr": float(np.mean(scores[name])),
                      "std_wsr": float(np.std(scores[name])),
                      "mean_cell_power": float(np.mean(powers[name])),
                      "max_cell_power": float(np.max(powers[name])),
                      "algorithm_seconds": seconds[name],
                      "seconds_per_sample": seconds[name] / size} for name in methods}
    result = {"data": str(args.data), "checkpoint": str(args.checkpoint),
              "samples": size, "device": str(device), "batch_size": args.batch_size,
              "fastfp_iterations": args.fastfp_iters, "wmmse_iterations": args.wmmse_iters,
              "seed": args.seed, "weights": weights,
              "power": 100, "noise": 1e-8,
              "score_scale": 6, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "method", "wsr", "cell_power"])
        writer.writeheader()
        writer.writerows(rows)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
