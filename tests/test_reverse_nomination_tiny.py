"""Smoke test for scripts/reverse_nomination.py on a tiny random Llama (CPU)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from test_shared_tiny import make_tiny_model  # noqa: E402
import reverse_nomination as rn  # noqa: E402


def test_scan_and_nominate():
    model = make_tiny_model()
    V = model.get_output_embeddings().weight.shape[0]
    logu = np.log(np.random.default_rng(0).dirichlet(np.ones(V)))
    df = rn.scan_layers(model, [1, 2], logu, k_null=4, chunk=50, verbose=False)
    n = model.config.intermediate_size
    assert len(df) == 2 * n and df.neuron_id.is_unique
    assert (df.w_norm > 0).all() and (df.logit_var > 0).all()
    assert ((df.nullfrac_k64 >= 0) & (df.nullfrac_k64 <= 1.0 + 1e-6)).all()
    assert df.freq_corr.abs().max() <= 1.0 + 1e-6 and df.freq_corr.notna().all()
    # logit_var is scale-invariant: doubling w_out leaves it unchanged -> check against manual computation
    W_U = model.get_output_embeddings().weight.detach().float()
    gamma = model.model.norm.weight.detach().float()
    w = gamma * model.model.layers[1].mlp.down_proj.weight.detach().float()[:, 3]
    lv = (W_U @ w).var(unbiased=True) / (w @ w)
    assert abs(float(lv) - df[df.neuron_id == "L1_N3"].logit_var.iloc[0]) < 1e-4
    ent, freq = rn.nominate(df, k=5, norm_quantile=0.5)
    assert len(ent) == 5 and len(freq) == 5 and not set(ent.neuron_id) & set(freq.neuron_id)
    assert (ent.w_norm >= df.w_norm.quantile(0.5)).all()
    assert ent.logit_var.max() <= df[df.w_norm >= df.w_norm.quantile(0.5)].logit_var.nsmallest(5).max() + 1e-12
    cand = rn.to_candidates(ent, {"note": "test"})
    assert set(cand) == {"provenance", "candidates"} and cand["candidates"][0]["nomination"] == "entropy_weights"


if __name__ == "__main__":
    test_scan_and_nominate(); print("ok")
