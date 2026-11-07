"""Three-cell DeepFP hybrid training demo."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from FP_unfolding.FastFP_net import FastFPnet
from FP_unfolding.loss_fuction import LossUnfolding
from FP_unfolding.compute_WSR import WSR2


ROOT = Path(__file__).resolve().parent
WEIGHTS = [i / 21 for i in range(1, 7)]


def load_split(data_dir, split, limit=0):
    """Read the existing three-cell HDF5 shards in manifest order."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    info = manifest["splits"][split]
    count = min(info["samples"], limit) if limit else info["samples"]
    channels = torch.empty((count, 3, 3, 6, 4, 64), dtype=torch.complex128)
    cursor = 0
    for shard in info["shards"]:
        if cursor == count:
            break
        take = min(shard["samples"], count - cursor)
        with h5py.File(data_dir / shard["file"], "r") as handle:
            data = handle["H"][:take]
        if data.shape != (take, 3, 3, 6, 4, 64) or data.dtype != np.complex128:
            raise ValueError(f"Unexpected channel data in {shard['file']}: {data.shape}, {data.dtype}")
        channels[cursor:cursor + take] = torch.from_numpy(data)
        cursor += take
    if cursor != count:
        raise ValueError(f"{split}: expected {count} channels, loaded {cursor}")
    return channels


def epoch_lr(epoch, args):
    if epoch < args.supervised_epochs:
        t = epoch / max(1, args.supervised_epochs - 1)
        return args.supervised_lr * (args.supervised_end_lr / args.supervised_lr) ** t
    t = (epoch - args.supervised_epochs) / max(1, args.unsupervised_epochs - 1)
    return args.unsupervised_lr * (args.unsupervised_end_lr / args.unsupervised_lr) ** t


def validate(model, channels, batch_size, seed):
    model.eval()
    scores = []
    with torch.no_grad():
        for offset in range(0, len(channels), batch_size):
            h = channels[offset:offset + batch_size].to(model.device)
            torch.manual_seed(seed + offset)
            init_v = model.init_input(len(h)).to(model.device)
            v, _ = model(h, action_noise_power=0, init_V=init_v)
            # WSR2 already sums over the three cells. We keep the original
            # code's normalized user weights and do not multiply by 6.
            scores.extend(WSR2(h, v, WEIGHTS, noise_power=1e-8).tolist())
    model.train()
    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data" / "three_cell")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "three_cell")
    parser.add_argument("--supervised-epochs", type=int, default=2)
    parser.add_argument("--unsupervised-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-iters", type=int, default=40)
    parser.add_argument("--supervised-samples", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--supervised-lr", type=float, default=0.005)
    parser.add_argument("--supervised-end-lr", type=float, default=0.001)
    parser.add_argument("--unsupervised-lr", type=float, default=1e-4)
    parser.add_argument("--unsupervised-end-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.supervised_epochs < 1 or args.unsupervised_epochs < 1:
        parser.error("Both training stages require at least one epoch")
    if min(args.supervised_lr, args.supervised_end_lr,
           args.unsupervised_lr, args.unsupervised_end_lr) <= 0:
        parser.error("Learning rates must be positive")
    device = torch.device("cuda:0" if (args.device == "auto" and torch.cuda.is_available())
                          or args.device == "cuda" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = load_split(args.data_dir, "train", args.train_limit)
    validation = load_split(args.data_dir, "validation", args.val_limit)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))
    supervised = train[indices[:min(args.supervised_samples, len(train))]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = FastFPnet(3, 6, 4, 64, WEIGHTS, 2, 8, device, hidden_size=64,
                      signal_max_power=100, noise_power=1e-8).to(device)
    criterion = LossUnfolding(signal_max_power=100, noise_power=1e-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.supervised_lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: epoch_lr(epoch, args) / args.supervised_lr)
    config = vars(args).copy()
    config.update({"data_dir": str(args.data_dir), "output_dir": str(args.output_dir),
                   "device": str(device), "train_samples": len(train),
                   "supervised_samples_used": len(supervised),
                   "validation_samples": len(validation), "cells": 3, "users_per_cell": 6,
                   "N_t": 64, "N_r": 4, "streams": 2, "layers": 8,
                   "hidden_size": 64, "weights": WEIGHTS, "power": 100,
                   "noise": 1e-8, "score_scale": 1,
                   "lr_schedule": "geometric per epoch within each stage"})
    (args.output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best = -math.inf
    started = time.time()
    total_epochs = args.supervised_epochs + args.unsupervised_epochs
    for epoch in range(total_epochs):
        stage = "supervised" if epoch < args.supervised_epochs else "unsupervised"
        stage_data = supervised if stage == "supervised" else train
        lr = optimizer.param_groups[0]["lr"]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(stage_data), batch_size=args.batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=0)
        losses = []
        for batch_idx, (h_cpu,) in enumerate(loader):
            h = h_cpu.to(device)
            init_v = model.init_input(len(h)).to(device)
            optimizer.zero_grad(set_to_none=True)
            v, _ = model(h, action_noise_power=0, init_V=init_v)
            if stage == "supervised":
                loss = criterion(h, v, WEIGHTS, V_init=init_v,
                                 mode="super", target_iters=args.target_iters)
            else:
                loss = criterion(h, v, WEIGHTS, mode="unsuper")
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}, batch {batch_idx+1}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(loader):
                print(f"epoch={epoch+1}/{total_epochs} stage={stage} "
                      f"batch={batch_idx+1}/{len(loader)} loss={losses[-1]:.6f} "
                      f"lr={lr:.8g} elapsed={time.time()-started:.1f}s", flush=True)
        val_mean, val_std = validate(model, validation, args.batch_size, args.seed + 1000000)
        row = {"epoch": epoch + 1, "stage": stage, "lr": lr,
               "train_loss": float(np.mean(losses)), "validation_wsr": val_mean,
               "validation_std": val_std, "elapsed_seconds": time.time() - started}
        history.append(row)
        torch.save(model.state_dict(), args.output_dir / "deepfp_hybrid_last.pth")
        if stage == "supervised" and epoch + 1 == args.supervised_epochs:
            torch.save(model.state_dict(), args.output_dir / "deepfp_supervised.pth")
        if stage == "unsupervised" and val_mean > best:
            best = val_mean
            torch.save(model.state_dict(), args.output_dir / "deepfp_hybrid_best.pth")
        with (args.output_dir / "training_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print("EPOCH_SUMMARY", json.dumps(row), flush=True)
        scheduler.step()
    print("BEST_VALIDATION", best, "seconds", time.time() - started, flush=True)


if __name__ == "__main__":
    main()
