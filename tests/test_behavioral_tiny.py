"""Smoke test for scripts/behavioral_test.py on a tiny random Llama (CPU)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from test_shared_tiny import make_tiny_model, make_tiny_tokenizer  # noqa: E402
import behavioral_test as bt  # noqa: E402


def _setup():
    model = make_tiny_model()
    tok = make_tiny_tokenizer()
    prompts = [f"{tok.bos_token} the answer is", f"{tok.bos_token} a b c the answer is", f"{tok.bos_token} c"]
    return model, tok, prompts


def test_generate_batch_clean_and_clamped_are_well_formed():
    model, tok, prompts = _setup()
    gens = bt.generate_batch(model, tok, prompts, max_new_tokens=4)
    assert len(gens) == len(prompts) and all(isinstance(g, str) for g in gens)
    layer, idx = model.config.num_hidden_layers - 1, 3
    hook = bt.ClampHook(layer, idx, 50.0)  # an absurd clamp must change something
    gens2 = bt.generate_batch(model, tok, prompts, max_new_tokens=4, hook=hook, layer_idx=layer)
    assert len(gens2) == len(prompts)
    # hook removed afterwards
    assert len(model.model.layers[layer].mlp.down_proj._forward_pre_hooks) == 0
    assert any(a != b for a, b in zip(gens, gens2))


def test_clamp_hook_only_touches_last_position_and_one_neuron():
    model, tok, prompts = _setup()
    layer, idx = 1, 5
    seen = {}

    def spy(module, args):
        seen["x"] = args[0].detach().clone()

    hook = bt.ClampHook(layer, idx, 7.0)
    dp = model.model.layers[layer].mlp.down_proj
    h1 = dp.register_forward_pre_hook(hook)
    h2 = dp.register_forward_pre_hook(spy)  # runs after the clamp
    try:
        enc = tok(prompts[:1], return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            model(**enc)
    finally:
        h1.remove(); h2.remove()
    x = seen["x"]
    assert torch.allclose(x[:, -1, idx], torch.tensor(7.0))
    assert not torch.allclose(x[:, :-1, idx], torch.tensor(7.0))  # earlier positions untouched


def test_activation_stats_and_summary():
    model, tok, prompts = _setup()
    neurons = [(1, 5), (2, 7)]
    st = bt.activation_stats(model, tok, prompts * 3, neurons, verbose=False)
    assert set(st) == set(neurons) and all(len(v) == 2 for v in st.values())
    rows = []
    for pid, ctrl in [("u0", False), ("u1", False), ("c0", True), ("c1", True)]:
        rows.append(dict(neuron_id="clean", condition="clean", prompt_id=pid, is_control=ctrl, clamp_value=np.nan,
                         hedged=(pid == "u0"), changed_vs_clean=False, edit_ratio=1.0, first_token="x"))
        rows.append(dict(neuron_id="L1_N5", condition="mean", prompt_id=pid, is_control=ctrl, clamp_value=0.1,
                         hedged=(not ctrl), changed_vs_clean=True, edit_ratio=0.5, first_token="y"))
    summ = bt.summarize(pd.DataFrame(rows))
    assert set(summ.arm) == {"uncertain", "control"}
    unc = summ[summ.arm == "uncertain"].iloc[0]
    assert unc.hedge_rate == 1.0 and unc.hedge_rate_clean == 0.5 and unc.n_gained_hedge == 1
    assert abs(unc.hedge_delta_interaction - 0.5) < 1e-9
    assert bt.HEDGE_RE.search("The answer is unknown to me") and not bt.HEDGE_RE.search("The answer is Paris")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
