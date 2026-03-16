# Required Data Structure

The scripts in this repository assume the following project layout.

```text
project_root/
├─ train_unet2.py
├─ train_e2_v46.py
├─ v15_copy.py
├─ v24_copy.py
├─ v14_MADDPG.py
├─ ga2.py
├─ models_unet/
│  ├─ scalers.pkl
│  ├─ unet_model_config.json
│  └─ ... trained model files ...
├─ conc/
│  ├─ base/
│  │  └─ base.csv
│  ├─ clean/
│  │  └─ clean.csv
│  └─ scenario_XXX_monthly_136_855_all_species.csv
├─ input_emi/
│  ├─ base/
│  │  ├─ AG.csv
│  │  ├─ AR.csv
│  │  ├─ IN.csv
│  │  ├─ PP.csv
│  │  └─ TR.csv
│  └─ scenario_XXX/
│     ├─ AG.csv
│     ├─ AR.csv
│     ├─ IN.csv
│     ├─ PP.csv
│     └─ TR.csv
├─ prov_grid_map/
│  └─ 36kmprov.csv
├─ health/
│  ├─ 36kmpop.csv
│  └─ Incidence.csv
└─ other data/
   ├─ cost.csv
   ├─ transport.csv
   ├─ region.csv
   ├─ GDP.csv
   └─ VSL.csv
```

## Key File Formats

## `prov_grid_map/36kmprov.csv`

- required columns: `FID`, `name`
- role: maps each grid cell to a province code

## `conc/base/base.csv` and `conc/clean/clean.csv`

- required columns:
  `FID`, `X`, `Y`, `HNO3`, `N2O5`, `NO2`, `HONO`, `NO`, `H2O2`, `O3`, `OH`, `FORM`, `ISOP`, `TERP`, `SO2`, `NH3`, `PM25_SO4`, `PM25_NO3`, `PM25_NH4`, `PM25_OC`, `PM25_TOT`
- role: baseline and clean scenario concentration fields on the full grid

## `input_emi/base/*.csv` and `input_emi/scenario_XXX/*.csv`

- example file: `input_emi/base/AG.csv`
- available columns in the current dataset:
  `lon`, `lat`, `co`, `nh3`, `no2`, `bc`, `pm25`, `pm10`, `so2`, `oc`, `voc`
- role: sector-specific gridded emissions

## `other data/cost.csv`

- required columns: `sheng`, `cost`, `cost2`, `cost3`, `a`, `b`, `species`
- role: province-level marginal abatement cost parameters

## `other data/transport.csv`

- format: square province-to-province matrix
- first column: source province
- remaining columns: receptor provinces
- role: cross-province transport used in the fairness reward

## `health/36kmpop.csv`

- required columns: `FID`, `pop`
- role: grid-level population used in health-benefit and coordination calculations

## `health/Incidence.csv`

- required columns: `FID`, `value`
- role: grid-level incidence rate used by the health module

## Release Advice

- If your raw scenario files are too large, release only the code and publish the dataset separately.
- If possible, provide a small demo subset so reviewers can run a smoke test.
- Keep file names stable, because several scripts currently use hard-coded relative paths.
