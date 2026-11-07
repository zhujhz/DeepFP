"""Single-cell DeepFP hybrid training with the original ascending user weights.

Stage 1 learns FastFP's 40-iteration solution from the same random starting V.
Stage 2 maximizes weighted sum-rate. Both stages use an explicit epoch scheduler.
"""
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


def load_channels(path, limit=0):
    with h5py.File(path, "r") as f:
        data = f["H"][:limit or None]
    h = torch.from_numpy(np.array(data, copy=True))
    if h.ndim != 6 or h.shape[1:] != (1, 1, 6, 4, 64) or h.dtype != torch.complex128:
        raise ValueError(f"Unexpected channel data in {path}: {h.shape}, {h.dtype}")
    return h


def epoch_lr(epoch, args):
    if epoch < args.supervised_epochs:
        t = epoch / max(1, args.supervised_epochs - 1)
        return args.supervised_lr * (args.supervised_end_lr / args.supervised_lr) ** t
    t = (epoch - args.supervised_epochs) / max(1, args.unsupervised_epochs - 1)
    return args.unsupervised_lr * (args.unsupervised_end_lr / args.unsupervised_lr) ** t


def validate(model, channels, batch_size, seed, weights):
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, len(channels), batch_size):
            h = channels[offset:offset + batch_size].to(model.device)
            torch.manual_seed(seed + offset)
            v0 = model.init_input(len(h))
            v, _ = model(h, action_noise_power=0, init_V=v0)
            values.extend((WSR2(h, v, weights, noise_power=1e-8) * 6).tolist())
    model.train()
    return float(np.mean(values)), float(np.std(values))


def main():
    parser = argparse.ArgumentParser(description="Train DeepFP on the provided single-cell split")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data" / "single_cell")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "single_cell")
    parser.add_argument("--supervised-epochs", type=int, default=2)
    parser.add_argument("--unsupervised-epochs", type=int, default=8)
    parser.add_argument("--supervised-samples", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--val-limit", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--target-iters", type=int, default=40)
    parser.add_argument("--supervised-lr", type=float, default=0.005)
    parser.add_argument("--supervised-end-lr", type=float, default=0.001)
    parser.add_argument("--unsupervised-lr", type=float, default=1e-4)
    parser.add_argument("--unsupervised-end-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    weights = WEIGHTS
    if args.supervised_epochs < 1 or args.unsupervised_epochs < 1:
        parser.error("Both training stages require at least one epoch")
    if min(args.supervised_lr, args.supervised_end_lr,
           args.unsupervised_lr, args.unsupervised_end_lr) <= 0:
        parser.error("Learning rates must be positive")
    device = torch.device("cuda:0" if (args.device == "auto" and torch.cuda.is_available())
                          or args.device == "cuda" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    train = load_channels(args.data_dir / "train.h5", args.train_limit)
    validation = load_channels(args.data_dir / "validation.h5", args.val_limit)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))
    supervised = train[indices[:min(args.supervised_samples, len(train))]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = FastFPnet(1, 6, 4, 64, weights, 2, 10, device, hidden_size=64,
                      signal_max_power=100, noise_power=1e-8).to(device)
    loss_fn = LossUnfolding(signal_max_power=100, noise_power=1e-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.supervised_lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: epoch_lr(epoch, args) / args.supervised_lr)
    config = vars(args).copy()
    config.update({"data_dir": str(args.data_dir), "output_dir": str(args.output_dir),
                   "device": str(device), "train_samples": len(train),
                   "supervised_samples_used": len(supervised), "validation_samples": len(validation),
                   "network_layers": 10, "hidden_size": 64, "weights": weights,
                   "signal_max_power": 100, "noise_power": 1e-8,
                   "lr_schedule": "geometric per epoch within each stage"})
    (args.output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best_validation = -math.inf
    started = time.time()
    total_epochs = args.supervised_epochs + args.unsupervised_epochs
    for epoch in range(total_epochs):
        stage = "supervised" if epoch < args.supervised_epochs else "unsupervised"
        stage_data = supervised if stage == "supervised" else train
        current_lr = optimizer.param_groups[0]["lr"]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(stage_data), batch_size=args.batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=0)
        losses = []
        for batch_index, (h_cpu,) in enumerate(loader):
            h = h_cpu.to(device)
            v0 = model.init_input(len(h))
            optimizer.zero_grad(set_to_none=True)
            v, _ = model(h, action_noise_power=0, init_V=v0)
            if stage == "supervised":
                loss = loss_fn(h, v, weights, V_init=v0, mode="super", target_iters=args.target_iters)
            else:
                loss = loss_fn(h, v, weights, mode="unsuper")
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}, batch {batch_index+1}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if batch_index == 0 or (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader):
                print(f"epoch={epoch+1}/{total_epochs} stage={stage} "
                      f"batch={batch_index+1}/{len(loader)} loss={losses[-1]:.6f} "
                      f"lr={current_lr:.8g} elapsed={time.time()-started:.1f}s", flush=True)
        val_mean, val_std = validate(model, validation, args.batch_size,
                                     args.seed + 1000000, weights)
        row = {"epoch": epoch + 1, "stage": stage, "lr": current_lr,
               "train_loss": float(np.mean(losses)), "validation_wsr": val_mean,
               "validation_std": val_std, "elapsed_seconds": time.time() - started}
        history.append(row)
        torch.save(model.state_dict(), args.output_dir / "deepfp_hybrid_last.pth")
        if stage == "supervised" and epoch + 1 == args.supervised_epochs:
            torch.save(model.state_dict(), args.output_dir / "deepfp_supervised.pth")
        if stage == "unsupervised" and val_mean > best_validation:
            best_validation = val_mean
            torch.save(model.state_dict(), args.output_dir / "deepfp_hybrid_best.pth")
        with (args.output_dir / "training_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print("EPOCH_SUMMARY", json.dumps(row), flush=True)
        scheduler.step()
    print("BEST_VALIDATION", best_validation, "seconds", time.time()-started, flush=True)


if __name__ == "__main__":
    main()
