# U-Net Training Manual

## Purpose

`U-net_trainging.py` trains a U-Net based surrogate model that maps emission changes and background chemical indicators to gridded `PM25_TOT` responses.

## Input Summary

The script expects:

- baseline concentration data in `conc/base/base.csv`
- clean-scenario concentration data in `conc/clean/clean.csv`
- baseline emission files in `input_emi/base/`
- scenario concentration files in `conc/`
- scenario emission files in `input_emi/scenario_*/`
- province-grid mapping in `prov_grid_map/36kmprov.csv`

See `DATA_STRUCTURE.md` for the exact directory layout.

## Core Settings

- Grid size: `120 x 144`
- Padded grid size: `128 x 144`
- Emission channels: `5 precursors x 5 sectors = 25`
- Chemical channels: `18 indicators x 2 scenarios = 36`
- Total input channels: `61`
- Output channel: `PM25_TOT`

## Run Training

```bash
python U-net_trainging.py --data_path ./ --output_path ./models_unet/ --pollutant PM25_TOT --epochs 1000 --batch_size 8 --k_folds 5 --load_scenarios 600 --normal_repeat 2 --extreme_repeat 10
```

## Important Arguments

- `--data_path`: project root containing `conc/`, `input_emi/`, and `prov_grid_map/`
- `--output_path`: directory for model checkpoints and logs
- `--pollutant`: target output variable, default is `PM25_TOT`
- `--epochs`: number of training epochs
- `--batch_size`: batch size
- `--k_folds`: number of folds for cross-validation; use `1` for a single train/validation split
- `--load_scenarios`: number of concentration/emission scenarios to load
- `--normal_repeat`: repetition factor for regular scenarios
- `--extreme_repeat`: repetition factor for boundary scenarios such as baseline and clean cases

## Outputs

The script writes results to `models_unet/`, including:

- trained model folders or checkpoints
- scaler files
- training logs
- cross-validation summaries
- training history CSV files

## Notes For Release

- Keep only the final training script and essential model metadata in the public repository.
- Large checkpoints and temporary figures should not be committed unless they are necessary for reproduction.
- If you release pre-trained weights, document the exact training command and the dataset version.
