# MAPPO Training Manual

## Scope

This project contains two MAPPO-related scripts:

- `train_e2_v46.py`: environment definition, reward components, health-effect utilities, and the original TensorFlow MAPPO baseline.
- `v15_copy.py`: main PyTorch MAPPO training script used for optimization experiments.
- `v24_copy.py`: region-priority MAPPO variant.

## Pipeline

1. Train the U-Net surrogate model with `train_unet2.py`.
2. Place the trained model files in `models_unet/`.
3. Prepare concentration, emission, cost, transport, health, and province-mapping data.
4. Launch MAPPO training.

## Environment Inputs

The RL environment is implemented in `train_e2_v46.py` as `RSMEmissionEnv`.

Required files:

- `models_unet/`
- `conc/base/base.csv`
- `conc/clean/clean.csv`
- `prov_grid_map/36kmprov.csv`
- `input_emi/base/`
- `other data/cost.csv`
- `other data/transport.csv`
- `other data/region.csv`
- `other data/GDP.csv`
- `health/36kmpop.csv`
- `health/Incidence.csv`

## Observation and Action Design

- Number of agents: one agent per province
- Action dimension: `25`
- Action meaning: emission-reduction factors for `5 precursors x 5 sectors`
- Max steps per episode: `8`
- Local observation dimension in the current setup: `action_dim + num_provinces + province_feature_dim`
- Global observation dimension in the current setup: `num_provinces * action_dim + num_provinces * 2 + 1`

## Reward Components

The reward design in `train_e2_v46.py` combines:

- PM2.5 target attainment
- emission-reduction cost
- health benefits
- coordination reward
- ranking or game-style competitive reward
- fairness term based on the transport matrix

## Run The Main PyTorch MAPPO Script

```bash
python v15_copy.py
```

Edit the configuration block near the bottom of the file to control:

- single-scenario or multi-scenario mode
- random seed
- province training order
- episode budget
- fine-tuning episodes
- parallel workers

## Run The Region-Priority MAPPO Variant

```bash
python v24_copy.py
```

## Original TensorFlow Baseline

```bash
python train_e2_v46.py
```

## Expected Outputs

Depending on the script, outputs include:

- trained actor and critic checkpoints
- TensorBoard logs
- reward curves
- per-scenario summary CSV files
- detailed training logs

## Release Recommendations

- Publicly release `train_e2_v46.py`, `v15_copy.py`, and `v24_copy.py` with cleaned English comments.
- Keep fixed experimental settings in a separate config file in future revisions.
- Separate environment code from training code if you plan a second open-source pass.
