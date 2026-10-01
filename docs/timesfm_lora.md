# Experiment C — Fine-tuned TimesFM 2.5 with LoRA

Experiment C evaluates parameter-efficient LoRA fine-tuning of the TimesFM 2.5 Transformers checkpoint. It uses the official Transformers and PEFT pathway, not the inference-only `timesfm` package API.

Each walk-forward fold creates a new adapter from the base checkpoint and trains only on canonical dengue case history through that fold's `train_end_period`. Validation and test cases are excluded from training. The adapter then forecasts the fold's validation and test origins. Training uses raw case counts because TimesFM 2.5 applies internal instance normalization; it does not reuse Experiment A's external `log1p` transform.

The fixed pilot configuration is deliberately small: 64-week context, 12-week horizon, LoRA rank 4, 2,000 random windows, and 5 epochs. It should be run first on fold 0, then expanded only if GPU memory and runtime are acceptable. Generated adapters are stored under `models/checkpoints/timesfm_lora/` and remain ignored by Git.

```powershell
.venv\Scripts\python.exe -m pip install --upgrade -r requirements-timesfm-finetune.txt
.venv\Scripts\python.exe scripts\training\49.timesfm_lora_finetune.py --dry-run
.venv\Scripts\python.exe scripts\training\49.timesfm_lora_finetune.py --folds 0 --suffix _pilot --no-holdout
.venv\Scripts\python.exe scripts\training\49.timesfm_lora_finetune.py
```

The arm writes point forecasts only. The established benchmark scorer will create its forecast intervals through its existing previous-year conformal calibration; it does not mislabel interpolated intervals as native LoRA uncertainty.

The implementation follows Google's documented [TimesFM 2.5 LoRA workflow](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning).
`timesfm-2.5-200m-transformers` uses the `timesfm2_5` architecture. The
requirements file pins Transformers 5.18.0 because the older 4.57 release does
not include it. Do not substitute the older `TimesFmModelForPrediction` class
because it targets a different architecture.

