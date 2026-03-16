# RL_paper
Code, documentation, and data related to the optimization of PM2.5 mitigation strategies in China using multi-agent reinforcement learning.
# Air Pollution Control With U-Net and Multi-Agent RL

This repository contains the code used to train a surrogate air-quality model and to optimize provincial emission-reduction strategies with multi-agent reinforcement learning and a genetic algorithm.

## Included Scripts

- `U-net_training.py`: train the U-Net based DeepRSM surrogate model.
- `MARL_Environment.py`: environment, reward design, and baseline TensorFlow MAPPO implementation.
- `MAPPO.py`: PyTorch MAPPO training script.
- `MAPPO-region.py`: PyTorch MAPPO training script with region-priority settings.
- `MADDPG.py`: PyTorch MADDPG training script.
- `GA.py`: genetic algorithm baseline for multi-step emission-reduction optimization.

## Recommended Open-Source Layout

The current codebase is research-oriented. For public release, keep only the files needed to reproduce the paper:

- `U-net_training.py`
- `MARL_Environment.py`
- `MAPPO.py`
- `MAPPO-region.py`
- `MADDPG.py`
- `GA.py`
- `README.md`
- `README_UNET.md`
- `README_MAPPO.md`
- `DATA_STRUCTURE.md`

Move large training outputs, local logs, and intermediate analysis results out of the repository or keep them ignored with `.gitignore`.

## Quick Start

1. Prepare the dataset using the directory layout in `DATA_STRUCTURE.md`.
2. Train the surrogate model with `U-net_training.py`.
3. Verify that `models_unet/` contains the trained model and scaler files.
4. Run one of the optimization scripts:
   - `MAPPO.py` for the main PyTorch MAPPO version.
   - `MAPPO-region.py` for the region-priority MAPPO variant.
   - `MADDPG.py` for the MADDPG baseline.
   - `GA.py` for the genetic algorithm baseline.

## Documentation

- `README_UNET.md`: U-Net training guide.
- `README_MAPPO.md`: MAPPO environment and training guide.
- `DATA_STRUCTURE.md`: required input files and expected folder layout.

## Citation

Add your paper citation here after publication.
