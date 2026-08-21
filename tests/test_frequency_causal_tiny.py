"""Smoke test for scripts/frequency_causal.py on a tiny random Llama (CPU)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from test_shared_tiny import make_tiny_model, make_tiny_tokenizer  # noqa: E402
import frequency_causal as fc  # noqa: E402


def test_dist_metrics_and_sweep():
    model, tok = make_tiny_model(), make_tiny_tokenizer()
    V = model.get_output_embeddings().weight.shape[0]
    logu = torch.log(torch.full((V,), 1.0 / V))  # uniform unigram
    p = torch.full((V,), 1.0 / V)
    ent, elogu, kl = fc.dist_metrics(p, logu)
    assert abs(ent - np.log(V)) < 1e-4 and abs(kl) < 1e-5 and abs(elogu - np.log(1 / V)) < 1e-5
    prompt = f"{tok.bos_token} the answer is"
    ms = fc.sweep_metrics(model, tok, prompt, 1, 3, [-1.0, 0.0, 1.0], logu)
    assert len(ms) == 3 and all(len(m) == 3 for m in ms)
    assert all(np.isfinite(x) for m in ms for x in m)
    assert len(model.model.layers[1].mlp.down_proj._forward_pre_hooks) == 0


def test_temp_matched_elogu_recovers_entropy():
    V = 50
    logits = torch.randn(V) * 3
    p = torch.softmax(logits, -1)
    logu = torch.log(torch.softmax(torch.randn(V), -1))
    target = 2.0
    eu_t, T = fc.temp_matched_elogu(p, target, logu)
    pt = torch.softmax(torch.log(p) / T, -1)
    ent = -(torch.xlogy(pt, pt)).sum().item()
    assert abs(ent - target) < 1e-3 and np.isfinite(eu_t)
    # T=1 when target equals the clean entropy
    ent0 = -(torch.xlogy(p, p)).sum().item()
    eu0, T0 = fc.temp_matched_elogu(p, ent0, logu)
    assert abs(T0 - 1.0) < 1e-2 and abs(eu0 - (p * logu).sum().item()) < 1e-3


def test_summarize_slopes_and_merge():
    rows = []
    for pid in ("u0", "u1", "u2"):
        for k in (-2, 0, 2):
            rows.append(dict(neuron_id="L1_N3", is_candidate=True, prompt_id=pid, is_control=False, sigma_level=k,
                             d_elogu=0.5 * k + 0.01, d_kl=-0.2 * k, d_entropy=0.1 * k))
    df = pd.DataFrame(rows)
    fs = pd.DataFrame([dict(neuron_id="L1_N3", freq_corr=0.4, abs_freq_corr_pctile=99.5)])
    s = fc.summarize(df, fs)
    r = s.iloc[0]
    assert abs(r.d_elogu_slope_per_sigma - 0.5) < 1e-9 and abs(r.d_kl_slope_per_sigma + 0.2) < 1e-9
    assert abs(r.d_elogu_mean_ablation - 0.01) < 1e-9 and r.freq_corr == 0.4


if __name__ == "__main__":
    test_dist_metrics_and_sweep(); test_summarize_slopes_and_merge(); print("ok")
