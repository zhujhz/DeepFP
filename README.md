# DeepFP experiments

The DeepFP implementation is in `FP_unfolding/`. The maintained demos train
and compare DeepFP, FastFP, and WMMSE for one cell or three cells. Each cell
has 6 users, 64 transmit antennas, 4 receive antennas, and 2 streams.
Both demos use transmit power 100 and code noise power `1e-8`.

| Demo | Data | Training | Testing | Outputs |
| --- | --- | --- | --- | --- |
| Single cell | `Data/single_cell/` | `train_singlecell.py` | `test_singlecell.py` | `models/single_cell/`, `results/single_cell/` |
| Three cells | `Data/three_cell/` | `train_threecell.py` | `test_threecell.py` | `models/three_cell/`, `results/three_cell/` |

Trained checkpoints are included in `models/single_cell/` and
`models/three_cell/`. The default test scripts load `deepfp_hybrid_best.pth`
from the corresponding folder. The large channel datasets are not yet included.
Place prepared HDF5 data in the corresponding `Data/` folders or generate it
as described below before testing or retraining.

## Install

```powershell
python -m pip install -r requirements.txt
```

Large MAT and HDF5 files under `Data/*/` are marked for Git LFS in
`.gitattributes`. Keeping both formats roughly doubles the channel-data
disk use; check free space before generating a large run.

## Generate and prepare data

If you have downloaded the prepared HDF5 data in `Data/single_cell/` and
`Data/three_cell/`, you can skip data generation and use the demos directly.
If you only need one scenario, its corresponding folder is sufficient.
Otherwise, generate the channel data as MAT files with MATLAB and convert
them to HDF5 with Python as described below.

`Data/generate_dataset.m` is the unified MATLAB entry point. The workflow
keeps **both** formats in separate directories under `Data/`:

```text
Data/raw_<run>/  MATLAB .mat shards + generated_manifest.json
Data/h5_<run>/   Python .h5 splits/shards + manifest.json
```

The generator accepts
the number of cells and users, antenna dimensions, train/validation/test
sizes, seed, bandwidth, cell spacing, shadowing standard deviation, output
directory, and filename template. Its default MAT directory is
`Data/raw_generated/`; choose a distinct `raw_<run>` directory for each run.
The filename template must contain `{split}` and `{part}`. For example,
from the repository root:

```matlab
addpath('Data');
generate_dataset('cells',1,'users_per_cell',6,'nt',64,'nr',4, ...
    'train_samples',35000,'validation_samples',500,'test_samples',10000, ...
    'samples_per_file',1000,'seed',20261105, ...
    'output_dir','Data/raw_single_cell_demo', ...
    'filename_template','{split}_{part}_N{count}_Nr{nr}_Nt{nt}_K{users}_BS{cells}_seed{seed}.mat');
```

The MATLAB command saves version 7.3 MAT shards containing complex `chn`
and a manifest in `Data/raw_single_cell_demo/`. Convert them with:

```powershell
python prepare_data.py generated --manifest Data/raw_single_cell_demo/generated_manifest.json
```

This writes `Data/h5_single_cell_demo/` by default. To choose another HDF5
directory, pass `--output-dir Data/h5_my_run`. For a new training run, pass
`--data-dir Data/h5_single_cell_demo` to the matching training script.
The Python converter reads the MAT files without changing or deleting them;
it refuses to write HDF5 into the MAT source directory or overwrite existing
HDF5 outputs. MATLAB also refuses to overwrite an existing generated run
unless `overwrite=true` is passed explicitly.

The two demo HDF5 datasets are in `Data/single_cell/` and
`Data/three_cell/`. To rebuild those historical HDF5 splits while
keeping both formats under `Data/`, put the numbered MAT files in separate
`Data/raw_single_cell_historical/` or `Data/raw_three_cell_historical/`
directories and run:

```powershell
python prepare_data.py single --source-dir Data/raw_single_cell_historical
python prepare_data.py three --source-dir Data/raw_three_cell_historical
```

These rebuild commands refuse to overwrite the currently saved HDF5 splits;
use a new `--output-dir` if those outputs already exist. For all modes, HDF5
stores complex128 `H` in Python order
`[sample, receiving_cell, transmitting_cell, user, Nr, Nt]`.

## Training

Run the single-cell or three-cell script from this directory:

```powershell
python train_singlecell.py
python train_threecell.py
```

Both use Adam, batch size 200, 5,000 supervised samples, FastFP targets
after 40 iterations, and seed `20261105`. Training has 2 supervised epochs
with learning rates `0.005` and `0.001`, followed by 8 unsupervised epochs
whose rates decrease geometrically from `1e-4` to `1e-5`. The scheduler
steps after each epoch. Single cell has 10 unfolding layers; three cells
have 8. Both use hidden width 64. The best unsupervised checkpoint is
selected by validation WSR.

These are script defaults. Use `--data-dir`, `--output-dir`,
`--supervised-lr`, `--supervised-end-lr`, `--unsupervised-lr`, and
`--unsupervised-end-lr` to override them. `train_config.json` and
`training_history.csv` record the settings and actual per-epoch rates.

## Testing and plots

```powershell
python test_singlecell.py
python test_threecell.py
python plot_training_curves.py
```

The single-cell test uses all 10,000 saved test channels by default and
100 iterations per baseline. The three-cell test uses 50 of 5,001 channels
and 30 iterations per baseline because its multicell WMMSE routine
processes channels individually. Both compare DeepFP with the existing
FastFP and WMMSE implementations using the same initial precoder.
`--limit` can select a smaller test set.
`comparison.json` stores summary and timings; `comparison.csv` stores
per-channel scores and power.

`plot_training_curves.py` reads the saved histories and writes PNG/SVG
training and loss curves under `results/<scenario>/`. Their x-axis is
elapsed training time in minutes; supervised MSE and unsupervised negative
WSR appear in separate panels.


