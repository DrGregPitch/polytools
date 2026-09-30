---
title: Polymer Tg Predictor
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Polymer Tg predictor — demo

Paste a polymer **repeat unit** as a pSMILES (an ordinary SMILES with two
connection points written `[*]`) and get a glass-transition temperature with a
**calibrated error bar** and an **applicability-domain** check. Built on
[`polytools`](https://github.com/DrGregPitch/polytools).

The demo deliberately shows three things, not one:

1. **Tg ± σ**, where σ is the ensemble spread recalibrated on held-out data — an
   error bar that means something, not decoration.
2. **Applicability domain** — the nearest-neighbour Tanimoto similarity of your
   repeat unit to the training set. A confident number on a polymer unlike anything
   in training is the exact trap this library exists to flag, so the demo says so
   out loud.
3. The **parsed repeat-unit structure**, so you can see what was actually modelled.

> The model is trained on a bundled 60-polymer toy set with approximate (and, for
> a few polymers, disputed) literature Tg values. **This demonstrates the
> pipeline; it is not a Tg oracle.** Point it at a model trained on clean
> experimental data for real predictions.

## Run locally

```bash
pip install -e ".[gbm]" gradio     # from the polytools repo root
python demo/app.py                 # opens a local Gradio server
```

## Deploy to a Hugging Face Space

1. Create a new **Gradio** Space on huggingface.co.
2. Copy `app.py` and `requirements.txt` from this folder into it. The
   `requirements.txt` already points at `github.com/DrGregPitch/polytools`, so the
   Space can `pip install` the package from your repo.
3. Push. The Space builds and serves automatically; link it from your main README.

## How it works

`app.py` keeps a clean separation: `predict_tg(psmiles)` is pure `polytools`
(testable, UI-free) and `build_demo()` wraps it in Gradio. The model is the GBM
ensemble from the benchmark, using the torch-free sklearn backend so the Space
stays light and never hits the LightGBM/PyTorch OpenMP clash. The recalibration
scale is fit once at startup from 5-fold out-of-fold predictions for stability.
