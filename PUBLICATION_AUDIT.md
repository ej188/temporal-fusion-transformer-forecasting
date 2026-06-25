# Public Release And Reorganization Audit

This repository should become a project-summary portfolio repo, not a full working dump of every experiment. The public structure should make it immediately clear where to find the TFT implementation, validation method, performance evaluation, and business/project impact.

## Executive Summary

Current tracked state:

- 2,153 tracked files.
- 87 notebooks.
- 444 tracked Python files, of which 416 are inside a committed virtual environment.
- 1,598 generated backtest PNGs across two result folders.
- 6 generated MAPE pie charts.
- 4 CatBoost-generated log/artifact files.
- About 297 MB under `ML/`, mostly generated plots and exploratory artifacts.
- Many notebooks contain embedded outputs, private data paths, supplier/part/facility identifiers, and exploratory scratch code.

Recommended public shape:

```text
README.md
PUBLICATION_AUDIT.md
.gitignore
requirements.txt

01_tft_model_implementation/
02_model_validation/
03_model_performance_evaluation/
04_project_impact/
05_baselines_and_context/
archive/
```

The four main public folders should each have one job:

- `01_tft_model_implementation/`: the TFT training pipeline and feature engineering.
- `02_model_validation/`: validation split, leakage prevention, metric definitions, and reproducibility notes.
- `03_model_performance_evaluation/`: cleaned evaluation notebook/script and summarized results.
- `04_project_impact/`: supply-chain interpretation, operational implications, and selected sanitized visuals.

## Current Folder Analysis

### Root

Keep and refine:

- `README.md`: now public-facing and focused on the TFT forecasting project.
- `.gitignore`: now covers common ML artifacts, private data folders, checkpoints, reports, logs, and environments.
- `PUBLICATION_AUDIT.md`: this file.

Recommended addition:

- `requirements.txt`: move or copy `ML/mj/requirements.txt` to root so reviewers can install dependencies without searching nested folders.

### `ML/Shrey/FE_ATTEMPTS/`

Purpose found: TFT implementation iterations.

Most important file:

- `FE_ATTEMPT5.py`: strongest public candidate. It includes leakage-safe lags/rolling features, supplier/part/global aggregates, TFT training, checkpointing, and validation metrics. It has been updated to accept `--data` instead of hard-coding an internal path.

Secondary candidates:

- `FE_ATTEMPT4.py`, `FE_ATTEMPT3.py`, `FE_ATTEMPT2.py`, `FE_ATTEMPT1.py`: useful only as development history. Do not put in the main public path.
- `FE_CULM.py`, `CLAW.py`, `thwin.py`, `lasttry2.py`, `lasttry3.py`: overlapping experimental TFT variants. Archive or delete.
- `Last62024.ipynb`: notebook output risk. Archive or delete after checking whether it contains a unique result.

Recommendation:

- Rename `FE_ATTEMPT5.py` to `01_tft_model_implementation/train_tft.py`.
- Extract reusable feature engineering and metric code later if you want a cleaner library structure, but do not block publication on that refactor.

### `ML/mj/sbatch/mj_TFT.py`

Purpose found: earlier modular TFT pipeline with a stronger function/class structure and feature registry.

Strengths:

- Has `Config`, `load_and_clean`, `engineer_features`, `filter_series`, `build_registry`, `build_datasets`, `create_model`, `train_model`, `evaluate`, and `visualize` functions.
- Good source for explaining the modeling workflow and separating validation/evaluation logic.

Risks:

- Still has local/default data paths and output assumptions.
- Includes richer feature engineering, but may be less directly tied to the final best result than `FE_ATTEMPT5.py`.

Recommendation:

- Rename to `01_tft_model_implementation/tft_pipeline_experimental.py` or archive as `archive/tft_iterations/mj_tft_pipeline.py`.
- If kept publicly, sanitize paths and add CLI arguments.

### `ML/Shrey/Other_stuff/actual_pred.ipynb`

Purpose found: strongest performance-evaluation artifact.

Strengths:

- Contains actual-vs-predicted comparison logic.
- Contains evaluation tables and presentation plot generation.
- Strong signal for model performance evaluation.

Risks:

- 2.7 MB notebook with 42 outputs.
- Contains private/internal data paths and likely printed supplier-part performance records.
- Should not be published as-is.

Recommendation:

- Convert into `03_model_performance_evaluation/evaluate_forecasts.py` or a stripped notebook named `03_model_performance_evaluation/performance_evaluation.ipynb`.
- Keep only aggregate metrics and sanitized plots.
- Remove all row-level supplier/part identifiers and notebook outputs.

### `MasterPlots/Plots/`

Purpose found: project-impact and operational visualization.

Potential impact files:

- `Worst Suppliers - Monthly Fill Rates.ipynb`: strongest impact artifact for explaining operational use.
- `Top 10 worst suppliers.ipynb`: good impact artifact, but title and outputs are sensitive and should be sanitized.
- `Months of History vs unique parts.ipynb`: useful data coverage/feasibility context.
- `Harini_plots.ipynb`, `Mohana_analyses.ipynb`: exploratory, review manually.

Recommendation:

- Do not publish these raw notebooks.
- Create sanitized summaries under `04_project_impact/`.
- Use neutral wording such as `supplier_fill_rate_analysis` instead of `worst suppliers`.

### `ML/regression-model/`

Purpose found: baseline regression and historical backtesting.

Keep only as context:

- `regression.ipynb`
- `time_series-regression-combined.ipynb`
- `(pivot)regression.ipynb`
- `(random)regression.ipynb`

Delete:

- `backtest-results/`: 796 generated PNGs, about 126 MB.
- `predicted_vs_actual_backtest_results/`: 796 generated PNGs, about 144 MB.

Recommendation:

- Keep a short baseline summary in `05_baselines_and_context/regression_baseline_summary.md`.
- Do not publish the full per-series generated image folders.

### `ML/mj/sbatch/SARIMAX/`

Purpose found: SARIMAX baseline and supplier fill-rate experiments.

Potential context files:

- `mj_SARIMAX.py`
- `sarimax_supply_chain_forecast (4).py`
- `sarimax_supply_chain_forecast (5).py`

Risks:

- File names contain copy/version numbers.
- May include internal paths and generated-output assumptions.

Recommendation:

- If you want baseline code, rename one cleaned file to `05_baselines_and_context/sarimax_baseline.py`.
- Otherwise summarize SARIMAX in markdown and archive/delete raw scripts.

### `ML/mj/sbatch/MMF/NeuralForecast/`

Purpose found: NHITS, NNF, SARIMAX, and NeuralForecast experiments.

Recommendation:

- Archive or delete for the public summary repo.
- If retained, put only one cleaned baseline script under `05_baselines_and_context/`.

### `ML/multi-model/`

Purpose found: multi-model exploration.

Recommendation:

- Archive unless it contributes directly to the public project narrative.
- Do not keep `Untitled.ipynb` publicly.

### `ML/catboost_info/`

Purpose found: generated CatBoost logs.

Recommendation:

- Delete. This is generated training metadata, not public source.

### `ML/encoder_cluster0.joblib`

Purpose found: serialized artifact.

Recommendation:

- Delete. Do not publish trained encoders or artifacts.

### `ML/old/` and `ML/super_old/`

Purpose found: old experiments, pivots, EDA, PyCaret, Kaggle, early time-series attempts.

Recommendation:

- Archive or delete from public repo.
- Do not leave folders named `old`, `super_old`, `previous`, or `Untitled` in a portfolio repo.

### `Code Review/`

Purpose found: team code-review notebooks and exploratory review outputs.

Risks:

- Internal paths.
- Supplier/part identifiers.
- Notebook outputs.
- Folder name contains a space.

Recommendation:

- Delete or archive. This does not belong in the main project-summary repo.
- If any insight is valuable, summarize it in `05_baselines_and_context/`.

### `DE/`

Purpose found: data engineering scratch notebooks and a TFT notebook.

Risks:

- Internal data paths.
- Notebooks with outputs.

Recommendation:

- Delete or archive.
- If data-preparation logic is needed, extract a sanitized schema/preprocessing description to `02_model_validation/data_schema.md` or `01_tft_model_implementation/preprocessing_notes.md`.

### `Gen_AI/`

Purpose found: slurm/job artifact, currently malformed as notebook JSON in a `.slurm` file.

Recommendation:

- Delete `Gen_AI/my_job.slurm`.

### `Semester_One/`

Purpose found: first-semester clustering, fine-tuning, Gen AI, and old environment work.

Critical delete:

- `Semester_One/fine_tuning/timesfm_venv/`: committed virtual environment with 416 tracked Python files.

Other recommendation:

- Archive or delete the rest for the public summary repo.
- Keep only public-facing summaries in `05_baselines_and_context/`, such as a cleaned version of `fine_tuning/Fine_Tuning_README.md` and `clustering/README.md`.

### `clustering/`

Purpose found: useful project context for segmentation and forecasting strategy.

Recommendation:

- Keep as context, but move/rename to `05_baselines_and_context/clustering_summary.md` or `04_project_impact/clustering_for_forecasting.md`.
- Remove explicit internal delivery references.

### `fine_tuning/`

Purpose found: model-selection and TimesFM/SARIMAX context.

Recommendation:

- Keep `fine_tuning/Fine_Tuning_README.md` as context after polishing.
- Delete or archive `fine_tuning/timesfm/` if it is copied external documentation rather than your own work.

## Required Public Folder Separation

### `01_tft_model_implementation/`

This folder should answer: "How was the TFT model implemented?"

Recommended contents:

```text
01_tft_model_implementation/
  README.md
  train_tft.py
  tft_pipeline_experimental.py
```

Source mapping:

```text
ML/Shrey/FE_ATTEMPTS/FE_ATTEMPT5.py -> 01_tft_model_implementation/train_tft.py
ML/mj/sbatch/mj_TFT.py -> 01_tft_model_implementation/tft_pipeline_experimental.py
```

Notes:

- `train_tft.py` should be the primary implementation.
- `tft_pipeline_experimental.py` should either be clearly labeled experimental or moved to archive.
- Long term, split `train_tft.py` into `features.py`, `train.py`, and `metrics.py`, but only if you want a more software-engineered repo.

### `02_model_validation/`

This folder should answer: "How did we validate the model and prevent leakage?"

Recommended contents:

```text
02_model_validation/
  README.md
  validation_strategy.md
  metric_definitions.md
  data_schema.md
```

Source material:

- Validation split and cutoff logic from `FE_ATTEMPT5.py`.
- Shifted lag/rolling feature logic from `FE_ATTEMPT5.py`.
- Baseline and metric logic from `FE_ATTEMPT5.py` and `mj_TFT.py`.

Recommended documentation topics:

- Train/validation cutoff: last 6 months held out.
- Encoder length and prediction horizon.
- Why all rolling/lag features use `shift(1)`.
- How WAPE, MAPE, MAE, and per-horizon MAE are computed.
- Why zero-demand series require guarded percentage metrics.

### `03_model_performance_evaluation/`

This folder should answer: "How well did the model perform?"

Recommended contents:

```text
03_model_performance_evaluation/
  README.md
  performance_summary.md
  evaluate_forecasts.py
  selected_figures/
```

Source mapping:

```text
ML/Shrey/Other_stuff/actual_pred.ipynb -> 03_model_performance_evaluation/performance_evaluation.ipynb
```

Required cleanup before keeping:

- Strip outputs.
- Remove internal paths.
- Remove supplier-part row-level identifiers.
- Replace any private file names with placeholders.
- Keep only aggregate metrics and sanitized visual summaries.

Recommended public metrics:

- WAPE: approximately 13.4% on the internal validation dataset.
- MAE overall.
- MAE by forecast horizon.
- MAPE with zero-demand protection.
- Example interpretation plots only if they are sanitized.

### `04_project_impact/`

This folder should answer: "Why did this project matter?"

Recommended contents:

```text
04_project_impact/
  README.md
  project_impact_summary.md
  supply_chain_use_cases.md
  selected_figures/
```

Source mapping:

```text
MasterPlots/Plots/Worst Suppliers - Monthly Fill Rates.ipynb -> 04_project_impact/supplier_fill_rate_trends.ipynb
MasterPlots/Plots/Top 10 worst suppliers.ipynb -> 04_project_impact/supplier_shortage_analysis.ipynb
MasterPlots/Plots/Months of History vs unique parts.ipynb -> 04_project_impact/data_coverage_analysis.ipynb
clustering/README.md -> 04_project_impact/clustering_for_forecasting.md
```

Required cleanup before keeping:

- Rename "worst suppliers" to neutral wording like `supplier_shortage_analysis`.
- Strip notebook outputs.
- Remove row-level identifiers.
- Use aggregate results, anonymized examples, or synthetic visuals.

Impact points to highlight:

- Better 6-month shipment planning.
- Better handling of intermittent and sparse demand.
- Ability to model many supplier-part series jointly.
- Reduced manual one-model-per-series forecasting burden.
- Feature interpretability through TFT variable selection/attention outputs.

### `05_baselines_and_context/`

This folder should answer: "What alternatives did we compare against?"

Recommended contents:

```text
05_baselines_and_context/
  README.md
  regression_baseline_summary.md
  sarimax_baseline_summary.md
  neuralforecast_experiments_summary.md
  fine_tuning_summary.md
```

Source material:

- `ML/regression-model/*.ipynb`
- `ML/mj/sbatch/SARIMAX/*.py`
- `ML/mj/sbatch/MMF/NeuralForecast/*.py`
- `fine_tuning/Fine_Tuning_README.md`

Recommendation:

- Prefer summaries over raw notebooks.
- Keep only cleaned scripts if they demonstrate a meaningful baseline.

### `archive/`

This folder should contain only material you intentionally keep for provenance. It should not be part of the main reviewer path.

Recommended contents:

```text
archive/
  tft_iterations/
  baseline_experiments/
  semester_one_notes/
```

Do not archive generated artifacts, private data, environments, or logs. Delete those instead.

## Concrete Rename Plan

Use `git mv` for files you keep so history is preserved.

```bash
mkdir -p 01_tft_model_implementation \
  02_model_validation \
  03_model_performance_evaluation/selected_figures \
  04_project_impact/selected_figures \
  05_baselines_and_context \
  archive/tft_iterations \
  archive/baseline_experiments \
  archive/semester_one_notes

git mv ML/Shrey/FE_ATTEMPTS/FE_ATTEMPT5.py 01_tft_model_implementation/train_tft.py
git mv ML/mj/sbatch/mj_TFT.py 01_tft_model_implementation/tft_pipeline_experimental.py
git mv ML/mj/requirements.txt requirements.txt
git mv clustering/README.md 04_project_impact/clustering_for_forecasting.md
git mv fine_tuning/Fine_Tuning_README.md 05_baselines_and_context/fine_tuning_summary.md
```

Optional after cleanup:

```bash
git mv ML/Shrey/Other_stuff/actual_pred.ipynb 03_model_performance_evaluation/performance_evaluation.ipynb
git mv "MasterPlots/Plots/Worst Suppliers - Monthly Fill Rates.ipynb" 04_project_impact/supplier_fill_rate_trends.ipynb
git mv "MasterPlots/Plots/Top 10 worst suppliers.ipynb" 04_project_impact/supplier_shortage_analysis.ipynb
git mv "MasterPlots/Plots/Months of History vs unique parts.ipynb" 04_project_impact/data_coverage_analysis.ipynb
```

Only perform the optional notebook moves after stripping outputs and sanitizing paths.

## Concrete Delete Plan

High-confidence deletes:

```bash
git rm -r Semester_One/fine_tuning/timesfm_venv \
  ML/regression-model/backtest-results \
  ML/regression-model/predicted_vs_actual_backtest_results \
  ML/time-series-model/mape_pies \
  ML/catboost_info \
  ML/encoder_cluster0.joblib \
  Gen_AI/my_job.slurm
```

Likely deletes or archive-only:

```text
Code Review/
DE/
ML/old/
ML/super_old/
ML/Shrey/old_slurms/
ML/Shrey/Other_stuff/unnesecary/
ML/multi-model/Untitled.ipynb
fine_tuning/timesfm/
Semester_One/
```

Do not delete these until you confirm they do not contain unique narrative material:

```text
MasterPlots/Plots/
ML/Shrey/Other_stuff/actual_pred.ipynb
ML/mj/sbatch/SARIMAX/
ML/mj/sbatch/MMF/NeuralForecast/
```

## Naming Conventions

Use lowercase, descriptive, underscore-separated names:

- Good: `train_tft.py`, `performance_evaluation.ipynb`, `supplier_fill_rate_trends.ipynb`.
- Avoid: `Untitled.ipynb`, `FE_ATTEMPT5.py`, `Other_stuff`, `unnesecary`, `super_old`, copied filenames with spaces or parentheses.

Folder names should be ordered and purpose-specific:

- `01_tft_model_implementation`
- `02_model_validation`
- `03_model_performance_evaluation`
- `04_project_impact`
- `05_baselines_and_context`

## Notebook Policy

Before any notebook is kept publicly:

```bash
jupyter nbconvert --clear-output --inplace path/to/notebook.ipynb
```

Then rerun these checks:

```bash
rg -n -uu --hidden -g '!/.git/**' -g '!**/*venv/**' \
  '(/anvil/projects|Cat_Data|/home/|mail-user|MY_EMAIL|cis220051|corporate/cat-ai)' .

rg -n -uu --hidden -g '!/.git/**' -g '!**/*venv/**' \
  '(api[_-]?key|secret|token|password|credential|bearer|private[_-]?key|client_secret|connection_string|s3://|gs://|azure|snowflake|databricks)' .
```

## Confidentiality And Quality Checklist

Before publishing:

- No raw CSV, Excel, Parquet, pickle, checkpoint, model-weight, or joblib artifacts are tracked.
- No virtual environments are tracked.
- No generated plot folders are tracked.
- No internal Anvil, home-directory, university/HPC, or private project paths remain.
- No email addresses, account IDs, or job-account metadata remain.
- No notebook outputs contain supplier IDs, part IDs, facility IDs, volumes, forecasts, or screenshots.
- The README explains the project outcome without redistributing private data.
- The main path through the repo is obvious in under one minute.

## Recommended Final Reviewer Path

A reviewer should be able to read in this order:

1. `README.md`
2. `01_tft_model_implementation/README.md`
3. `01_tft_model_implementation/train_tft.py`
4. `02_model_validation/validation_strategy.md`
5. `03_model_performance_evaluation/performance_summary.md`
6. `04_project_impact/project_impact_summary.md`
7. `05_baselines_and_context/README.md`

That path cleanly separates implementation, validation, performance, and impact, which is the main goal of the public repo.
