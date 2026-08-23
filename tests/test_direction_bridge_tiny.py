"""Smoke test for scripts/direction_bridge.py core functions on a tiny random Llama (CPU)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from test_shared_tiny import make_tiny_model, make_tiny_tokenizer  # noqa: E402
import direction_bridge as db  # noqa: E402


def test_residuals_directions_cosines_and_steering():
    model, tok = make_tiny_model(), make_tiny_tokenizer()
    prompts = [f"{tok.bos_token} a b the answer is", f"{tok.bos_token} c the answer is", f"{tok.bos_token} b c a", f"{tok.bos_token} a"]
    layers = [1, 2]
    X = db.last_token_residuals(model, tok, prompts, layers, verbose=False)
    d = model.config.hidden_size
    assert X[1].shape == (4, d) and X[2].shape == (4, d)
    mask_a = np.array([True, True, False, False])
    dirn = db.diff_of_means(X[2], mask_a, ~mask_a)
    assert abs(float(dirn.norm()) - 1.0) < 1e-5
    cos = db.neuron_cosines(model, layers, dirn, "cpu")
    assert len(cos) == 2 * model.config.intermediate_size and cos.cos.abs().max() <= 1.0 + 1e-5
    # manual check of one cosine
    gamma = model.model.norm.weight.detach().float()
    w = gamma * model.model.layers[2].mlp.down_proj.weight.detach().float()[:, 5]
    manual = float((w @ dirn) / (w.norm() * dirn.norm()))
    assert abs(manual - float(cos[cos.neuron_id == "L2_N5"].cos.iloc[0])) < 1e-5
    recs = [dict(prompt_id=f"p{i}", chat_formatted_prompt=p, is_control=(i % 2 == 1)) for i, p in enumerate(prompts)]
    hedge_ids, answer_ids = [1, 2], [3, 4]
    n_hooks_before = len(model.model.layers[2]._forward_hooks)  # transformers>=5 may keep its own capture hook here
    s = db.steer_readout(model, tok, recs, 2, dirn, [-1.0, 0.0, 1.0], hedge_ids, answer_ids, sigma=2.0)
    assert len(s) == 12 and np.isfinite(s.hedge_logodds).all() and np.isfinite(s.entropy).all()
    base = s[s.alpha == 0.0].set_index("prompt_id").hedge_logodds
    moved = s[s.alpha == 1.0].set_index("prompt_id").hedge_logodds
    assert (base != moved).any()  # steering changes the readout
    assert len(model.model.layers[2]._forward_hooks) == n_hooks_before  # our hook removed (library hooks untouched)


if __name__ == "__main__":
    test_residuals_directions_cosines_and_steering(); print("ok")
