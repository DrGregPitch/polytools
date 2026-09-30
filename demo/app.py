"""Clickable Tg demo: paste a polymer repeat unit, get Tg with honest error bars.

    python demo/app.py            # launches a local Gradio server

The point of a demo in a portfolio is not the number; it is showing you know a
prediction without an uncertainty and an applicability check is not a prediction a
lab can use. So this returns three things, not one:

* a Tg estimate with a **calibrated** +/- (the ensemble sigma, recalibrated on a
  held-out split -- see polytools.metrics),
* an **applicability-domain** flag: how similar is this repeat unit to anything the
  model trained on? A confident-looking number on a polymer unlike everything in
  training is exactly the trap this whole library exists to avoid,
* the parsed **repeat-unit structure**, so you can see what was actually modelled.

The model is the GBM ensemble from the benchmark, trained on the bundled 60-polymer
toy set, so the demo runs offline in seconds and needs no GPU or torch. **The toy
labels are approximate and several are disputed -- this demonstrates the pipeline,
it is not a Tg oracle.** Point it at a real trained model for real predictions.
"""

from __future__ import annotations

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.model_selection import KFold

from polytools import (
    EnsembleRegressor,
    GBMRegressor,
    SigmaRecalibrator,
    featurize,
    is_valid_psmiles,
    load_toy_tg,
    prepare_mol,
    silence_rdkit,
)

silence_rdkit()

# --- train once at startup -------------------------------------------------
_DS = load_toy_tg()
_X, _ = featurize(_DS.psmiles, "ecfp+desc", n_bits=1024)


def _make_ensemble() -> EnsembleRegressor:
    return EnsembleRegressor(
        factory=lambda i: GBMRegressor(
            n_estimators=400, random_state=i, quantile_uncertainty=False,
            backend="sklearn",  # torch-free demo; sklearn HistGBM has no OMP clash
        ),
        n_members=5, bootstrap=True, include_member_sigma=False,
    )


# Estimate a *stable* recalibration scale from out-of-fold predictions over the
# whole set, rather than from one tiny validation split (which on 60 points is too
# noisy and produces wild error bars). Then train the final model on all the data.
_oof_mu = np.full(len(_DS), np.nan)
_oof_sd = np.full(len(_DS), np.nan)
for _tr, _te in KFold(n_splits=5, shuffle=True, random_state=0).split(_X):
    _m = _make_ensemble().fit(_X[_tr], _DS.y[_tr])
    _oof_mu[_te], _oof_sd[_te] = _m.predict(_X[_te], return_std=True)
_RECAL = SigmaRecalibrator().fit(_DS.y, _oof_mu, _oof_sd)

_MODEL = _make_ensemble().fit(_X, _DS.y)

# fingerprints of the whole set, for the applicability-domain check
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_TRAIN_FPS = [
    _GEN.GetFingerprint(prepare_mol(ps, mode="oligomer")) for ps in _DS.psmiles
]
_TRAIN_NAMES = _DS.names


def nearest_training_neighbour(psmiles: str) -> tuple[float, str]:
    """Max Tanimoto similarity of this repeat unit to the training set, and its name."""
    fp = _GEN.GetFingerprint(prepare_mol(psmiles, mode="oligomer"))
    sims = DataStructs.BulkTanimotoSimilarity(fp, _TRAIN_FPS)
    j = int(np.argmax(sims))
    name = str(_TRAIN_NAMES[j]) if _TRAIN_NAMES is not None else "a training polymer"
    return float(sims[j]), name


def predict_tg(psmiles: str) -> dict:
    """Core prediction, independent of any UI. Returns a dict of results.

    Keys: ``ok`` (bool), and on success ``tg``, ``sigma``, ``nn_similarity``,
    ``nn_name``, ``domain`` (in-domain / borderline / out-of-domain), ``message``;
    on failure ``error``.
    """
    psmiles = (psmiles or "").strip()
    if not is_valid_psmiles(psmiles):
        return {
            "ok": False,
            "error": (
                "Not a valid polymer repeat unit. It needs exactly two "
                "connection points, written [*] or *. Example: [*]CC([*])c1ccccc1"
            ),
        }

    X = featurize([psmiles], "ecfp+desc", n_bits=1024)[0]
    mu, sd_raw = _MODEL.predict(X, return_std=True)
    sd = _RECAL.transform(sd_raw)
    tg, sigma = float(mu[0]), float(sd[0])

    nn_sim, nn_name = nearest_training_neighbour(psmiles)
    if nn_sim >= 0.6:
        domain = "in-domain"
    elif nn_sim >= 0.4:
        domain = "borderline"
    else:
        domain = "out-of-domain"

    return {
        "ok": True,
        "tg": tg,
        "sigma": sigma,
        "nn_similarity": nn_sim,
        "nn_name": nn_name,
        "domain": domain,
    }


def structure_image(psmiles: str):
    """A PIL image of the parsed repeat unit (oligomer view), or None if unparseable."""
    try:
        mol = prepare_mol(psmiles, mode="oligomer", n_repeat=2)
        return Draw.MolToImage(mol, size=(420, 260))
    except Exception:
        return None


def _format(result: dict) -> str:
    if not result["ok"]:
        return f"⚠️ {result['error']}"
    domain_note = {
        "in-domain": "similar to polymers the model has seen — most trustworthy here.",
        "borderline": "only loosely similar to the training set — treat with caution.",
        "out-of-domain": (
            "unlike anything in training — the model is **extrapolating**, and the "
            "error bar likely understates the true uncertainty."
        ),
    }[result["domain"]]
    return (
        f"## Tg ≈ {result['tg']:.0f} ± {result['sigma']:.0f} °C\n\n"
        f"**Applicability domain:** {result['domain']} "
        f"(nearest training polymer: *{result['nn_name']}*, "
        f"Tanimoto {result['nn_similarity']:.2f}).\n\n"
        f"{domain_note}\n\n"
        f"---\n*Trained on a 60-polymer toy set with approximate labels — a "
        f"pipeline demo, not a Tg oracle.*"
    )


def _predict_ui(psmiles: str):
    """Gradio callback: returns (markdown, image)."""
    result = predict_tg(psmiles)
    return _format(result), structure_image(psmiles) if result["ok"] else None


def build_demo():
    """Construct the Gradio interface (imported lazily so the core stays UI-free)."""
    import gradio as gr

    examples = [
        ["[*]CC([*])c1ccccc1"],          # polystyrene
        ["[*]CC[*]"],                     # polyethylene
        ["[*][Si](C)(C)O[*]"],            # PDMS
        ["[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"],  # PET
        ["[*]CC(C)(C(=O)OC)[*]"],         # PMMA
    ]
    with gr.Blocks(title="Polymer Tg predictor") as demo:
        gr.Markdown(
            "# Polymer glass-transition (Tg) predictor\n"
            "Paste a **repeat unit** as a pSMILES — an ordinary SMILES with two "
            "connection points written `[*]`. You get a Tg with a calibrated error "
            "bar and an honest applicability-domain check."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Repeat unit (pSMILES)",
                                 value="[*]CC([*])c1ccccc1")
                btn = gr.Button("Predict Tg", variant="primary")
                gr.Examples(examples, inputs=inp)
            with gr.Column():
                out_md = gr.Markdown()
                out_img = gr.Image(label="Parsed repeat unit", type="pil")
        btn.click(_predict_ui, inputs=inp, outputs=[out_md, out_img])
        inp.submit(_predict_ui, inputs=inp, outputs=[out_md, out_img])
    return demo


if __name__ == "__main__":
    build_demo().launch()
