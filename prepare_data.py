"""Prepare DeepFP HDF5 splits from either historical or newly generated MAT files.

Subcommands:
  single     Recreate the recorded single-cell file-level split.
  three      Recreate the recorded three-cell random split.
  generated  Convert generate_dataset.m output using generated_manifest.json.
             Source MAT files remain untouched; HDF5 goes to Data/h5_<run>.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SINGLE_SHAPE = (1, 1, 6, 4, 64)
THREE_SHAPE = (3, 3, 6, 4, 64)
HISTORICAL_BLOCK_SIZE = 5000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_outputs(paths):
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output: " + ", ".join(existing))


def convert_raw(raw, shape, path):
    if raw.dtype.fields and {"real", "imag"} <= set(raw.dtype.fields):
        channels = raw["real"] + 1j * raw["imag"]
    else:
        channels = raw
    # h5py reverses MATLAB dimensions and suppresses trailing singleton axes.
    if shape[0] == 1:
        if channels.ndim != 4:
            raise ValueError(f"Unexpected single-cell MAT shape in {path}: {channels.shape}")
        channels = channels.transpose(3, 0, 2, 1)[:, None, None]
    else:
        if channels.ndim != 6:
            raise ValueError(f"Unexpected multicell MAT shape in {path}: {channels.shape}")
        channels = channels.transpose(5, 0, 1, 2, 4, 3)
    if channels.shape[1:] != shape:
        raise ValueError(f"Unexpected channel shape in {path}: {channels.shape}")
    return np.asarray(channels, dtype=np.complex128)


def read_mat(path, start=None, stop=None):
    with h5py.File(path, "r") as handle:
        return handle["chn"][..., slice(start, stop)]


def prepare_single(args):
    groups = {"train": range(1, 8), "validation": (8,), "test": (9, 10)}
    sizes = {"train": 35000, "validation": 500, "test": 10000}
    source = {i: args.source_dir / f"{i}_5000_Nr_4_Nt_64_N_user_6_BS_1.mat"
              for i in range(1, 11)}
    for path in source.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    check_outputs([args.output_dir / f"{split}.h5" for split in groups] +
                  [args.output_dir / "manifest.json"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"shape_convention": "samples,receiving_BS,transmitting_BS,users,Nr,Nt",
                "dtype": "complex128", "splits": {}}
    for split, blocks in groups.items():
        dest = args.output_dir / f"{split}.h5"
        cursor = 0
        inputs = []
        with h5py.File(dest, "w") as output:
            h = output.create_dataset("H", (sizes[split], *SINGLE_SHAPE),
                                      dtype=np.complex128, chunks=(100, *SINGLE_SHAPE))
            for number in blocks:
                path = source[number]
                block = convert_raw(read_mat(path), SINGLE_SHAPE, path)
                if split == "validation":
                    block = block[:500]
                h[cursor:cursor + len(block)] = block
                cursor += len(block)
                inputs.append({"name": path.name, "sha256": sha256(path)})
                print(split, path.name, cursor, flush=True)
            if cursor != sizes[split]:
                raise ValueError(f"{split}: expected {sizes[split]}, got {cursor}")
            output.attrs["split"] = split
            output.attrs["channel_shape"] = "N,1,1,6,4,64"
        manifest["splits"][split] = {"file": dest.name, "samples": cursor,
                                      "source_files": inputs, "sha256": sha256(dest)}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def historical_three_source(source_dir, block):
    return source_dir / f"{block}_5000_Nr_4_Nt_64_N_user_6_BS_3.mat"


def export_historical_three_shard(dest, global_indices, source_dir):
    count = len(global_indices)
    with h5py.File(dest, "w") as output:
        target = output.create_dataset("H", (count, *THREE_SHAPE),
                                       dtype=np.complex128,
                                       chunks=(min(10, count), *THREE_SHAPE))
        for block in range(1, 11):
            positions = np.flatnonzero(global_indices // HISTORICAL_BLOCK_SIZE == block - 1)
            if not len(positions):
                continue
            local_indices = global_indices[positions] % HISTORICAL_BLOCK_SIZE
            order = np.argsort(local_indices)
            path = historical_three_source(source_dir, block)
            with h5py.File(path, "r") as source:
                raw = source["chn"][..., local_indices[order]]
            channels = convert_raw(raw, THREE_SHAPE, path)
            target[positions] = channels[np.argsort(order)]
        output.attrs["shape_convention"] = (
            "samples,receiving_BS,transmitting_BS,users,Nr,Nt")


def prepare_three(args):
    sources = [historical_three_source(args.source_dir, i) for i in range(1, 11)]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    total = 10 * HISTORICAL_BLOCK_SIZE
    permutation = torch.randperm(total,
                                  generator=torch.Generator().manual_seed(args.seed)).numpy()
    train_count = int(0.8 * total)
    # Preserve the original split_matrix floating-point calculation: 4999.
    val_count = int(((1 - 0.8) / 2) * total)
    splits = {"train": permutation[:train_count],
              "validation": permutation[train_count:train_count + val_count],
              "test": permutation[train_count + val_count:]}
    names = [args.output_dir / "manifest.json"]
    for split, indices in splits.items():
        names.append(args.output_dir / f"{split}_indices.npy")
        names.extend(args.output_dir / f"{split}_{part:03d}.h5"
                     for part in range((len(indices) + args.shard_size - 1) // args.shard_size))
    check_outputs(names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "source_total": total,
                "shape_convention": "samples,receiving_BS,transmitting_BS,users,Nr,Nt",
                "dtype": "complex128", "shard_size": args.shard_size,
                "split_method": "torch.randperm then split_matrix(train_size=0.8)",
                "source_files": [{"file": p.name, "sha256": sha256(p)} for p in sources],
                "splits": {}}
    for split, indices in splits.items():
        index_path = args.output_dir / f"{split}_indices.npy"
        np.save(index_path, indices)
        shards = []
        for start in range(0, len(indices), args.shard_size):
            dest = args.output_dir / f"{split}_{start // args.shard_size:03d}.h5"
            part = indices[start:start + args.shard_size]
            export_historical_three_shard(dest, part, args.source_dir)
            shards.append({"file": dest.name, "samples": len(part), "sha256": sha256(dest)})
            print(split, dest.name, len(part), flush=True)
        manifest["splits"][split] = {
            "samples": len(indices), "index_file": index_path.name,
            "index_sha256": sha256(index_path), "shards": shards}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def prepare_generated(args):
    source_dir = args.manifest.parent.resolve()
    if args.output_dir is None:
        run_name = source_dir.name.removeprefix("raw_")
        args.output_dir = ROOT / "Data" / f"h5_{run_name}"
    if args.output_dir.resolve() == source_dir:
        raise ValueError("HDF5 output directory must differ from the MAT source directory")
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if source_manifest.get("format_version") != 1:
        raise ValueError("Unsupported generated manifest format")
    cfg = source_manifest["config"]
    cells = int(cfg["cells"])
    shape = (cells, cells, int(cfg["users_per_cell"]), int(cfg["nr"]), int(cfg["nt"]))
    groups = {}
    destinations = [args.output_dir / "manifest.json"]
    for split in ("train", "validation", "test"):
        files = sorted((item for item in source_manifest["files"]
                        if item["split"] == split), key=lambda item: item["part"])
        count = int(cfg[f"{split}_samples"])
        if sum(item["samples"] for item in files) != count:
            raise ValueError(f"Manifest sample count mismatch for {split}")
        if [item["part"] for item in files] != list(range(len(files))):
            raise ValueError(f"Missing or duplicate part number for {split}")
        groups[split] = (files, count)
        if cells == 1 and count:
            destinations.append(args.output_dir / f"{split}.h5")
        elif cells > 1:
            destinations.extend(args.output_dir / f"{split}_{part:03d}.h5"
                                for part in range((count + args.shard_size - 1) // args.shard_size))
        for item in files:
            path = args.manifest.parent / item["name"]
            if not path.is_file():
                raise FileNotFoundError(path)
    check_outputs(destinations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source_manifest": str(args.manifest.resolve()),
              "shape_convention": "samples,receiving_BS,transmitting_BS,users,Nr,Nt",
              "dtype": "complex128", "config": cfg, "splits": {}}
    for split, (files, count) in groups.items():
        if count == 0:
            report["splits"][split] = {"samples": 0, "shards": []}
            continue
        shard_size = count if cells == 1 else args.shard_size
        source_idx = source_offset = 0
        shards = []
        for part, start in enumerate(range(0, count, shard_size)):
            shard_count = min(shard_size, count - start)
            name = f"{split}.h5" if cells == 1 else f"{split}_{part:03d}.h5"
            with h5py.File(args.output_dir / name, "w") as output:
                h = output.create_dataset("H", (shard_count, *shape),
                                          dtype=np.complex128,
                                          chunks=(min(args.read_batch, shard_count), *shape))
                cursor = 0
                while cursor < shard_count:
                    source = files[source_idx]
                    available = source["samples"] - source_offset
                    take = min(args.read_batch, shard_count - cursor, available)
                    path = args.manifest.parent / source["name"]
                    block = convert_raw(read_mat(path, source_offset,
                                                 source_offset + take), shape, path)
                    h[cursor:cursor + take] = block
                    cursor += take
                    source_offset += take
                    if source_offset == source["samples"]:
                        source_idx += 1
                        source_offset = 0
                output.attrs["split"] = split
                output.attrs["channel_shape"] = str(shape)
            shards.append({"file": name, "samples": shard_count})
            print(split, name, shard_count, flush=True)
        if cells == 1:
            report["splits"][split] = {"file": shards[0]["file"], "samples": count}
        else:
            report["splits"][split] = {"samples": count, "shards": shards}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    single = subcommands.add_parser("single", help="Recreate historical single-cell split")
    single.add_argument("--source-dir", type=Path, required=True)
    single.add_argument("--output-dir", type=Path, default=ROOT / "Data" / "single_cell")
    three = subcommands.add_parser("three", help="Recreate historical three-cell split")
    three.add_argument("--source-dir", type=Path, required=True)
    three.add_argument("--output-dir", type=Path, default=ROOT / "Data" / "three_cell")
    three.add_argument("--seed", type=int, default=20261104)
    three.add_argument("--shard-size", type=int, default=5000)
    generated = subcommands.add_parser("generated", help="Convert generate_dataset.m MAT files")
    generated.add_argument("--manifest", type=Path, required=True)
    generated.add_argument("--output-dir", type=Path,
                           help="HDF5 directory (default Data/h5_<MAT directory name without raw_>)")
    generated.add_argument("--shard-size", type=int, default=5000)
    generated.add_argument("--read-batch", type=int, default=100)
    args = parser.parse_args()
    if hasattr(args, "shard_size") and args.shard_size < 1:
        parser.error("--shard-size must be positive")
    if hasattr(args, "read_batch") and args.read_batch < 1:
        parser.error("--read-batch must be positive")
    {"single": prepare_single, "three": prepare_three,
     "generated": prepare_generated}[args.mode](args)


if __name__ == "__main__":
    main()
