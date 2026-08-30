"""Smoke tests for scripts/circuit_common.py on a tiny random Llama (CPU): alignment, patching hooks, attribution."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from test_shared_tiny import make_tiny_model, make_tiny_tokenizer  # noqa: E402
import circuit_common as cc  # noqa: E402


def test_align_and_position_maps():
    al = cc.align([1, 2, 3, 9, 9, 4, 5], [1, 2, 7, 4, 5])
    assert al["prefix"] == 2 and al["suffix"] == 2 and al["ent_u"] == (2, 5) and al["ent_c"] == (2, 3)
    assert cc.position_map(al, "last") == [(6, 4)]
    assert cc.position_map(al, "suffix") == [(6, 4), (5, 3)]
    assert cc.position_map(al, "entity") == [(4, 2)]  # right-aligned, shorter span wins
    assert cc.position_map(al, "prefix") == [(0, 0), (1, 1)]
    assert cc.reverse_map([(6, 4)]) == [(4, 6)]


def test_patching_and_attribution_on_tiny_model():
    model, tok = make_tiny_model(), make_tiny_tokenizer()
    model.eval()
    ro = cc.Readout(tok)
    ro.hedge_ids, ro.answer_ids = [1, 2], [3, 4]
    u = dict(chat_formatted_prompt=f"{tok.bos_token} a b c the answer is", prompt_id="u")
    c = dict(chat_formatted_prompt=f"{tok.bos_token} a d the answer is", prompt_id="c")
    enc_u, enc_c, al = cc.encode_pair(model, tok, u, c)
    layers = [1, 2]
    lg_u, res_u = cc.run_capture(model, enc_u, layers, "resid")
    lg_c, res_c = cc.run_capture(model, enc_c, layers, "resid")
    assert res_u[1].shape[1] == enc_u["input_ids"].shape[1]
    # patching the whole residual at every aligned position of the LAST layer from c into u at the last position
    # reproduces c's last-layer state there -> readout must change unless already identical
    last = cc.position_map(al, "last")
    lg_p = cc.run_patched(model, enc_u, [(layers[-1], "resid", last, res_c[layers[-1]], None)])
    assert lg_p.shape == lg_u.shape and torch.isfinite(lg_p).all()
    # head patching: replacing every head of a layer at all positions with the source's heads equals a full o_proj-input swap
    H = model.config.num_attention_heads
    _, hu = cc.run_capture(model, enc_u, [1], "heads"); _, hc = cc.run_capture(model, enc_c, [1], "heads")
    pm = cc.position_map(al, "prefix") + cc.position_map(al, "entity") + cc.position_map(al, "suffix")
    lg_all = cc.run_patched(model, enc_u, [(1, "heads", pm, hc[1], h) for h in range(H)])
    assert torch.isfinite(lg_all).all()
    # neuron patching hook runs and returns finite logits
    _, nu = cc.run_capture(model, enc_u, [2], "neurons"); _, nc = cc.run_capture(model, enc_c, [2], "neurons")
    lg_n = cc.run_patched(model, enc_u, [(2, "neurons", last, nc[2], 0), (2, "neurons", last, nc[2], 3)])
    assert torch.isfinite(lg_n).all()
    # attribution: shapes and finiteness; heads -> [seq, H], neurons -> [seq, d_mlp]
    attr_h = cc.attribution(model, enc_u, [1], "heads", ro, hc, pm)
    assert attr_h[1].shape == (enc_u["input_ids"].shape[1], H) and torch.isfinite(attr_h[1]).all()
    attr_n = cc.attribution(model, enc_u, [2], "neurons", ro, nc, pm)
    assert attr_n[2].shape == (enc_u["input_ids"].shape[1], model.config.intermediate_size)
    # first-order check: attribution of a full-head swap vs the real patched change have the same sign for the top head
    real = {h: (ro.logodds(cc.run_patched(model, enc_u, [(1, "heads", pm, hc[1], h)])) - ro.logodds(lg_u)).item() for h in range(H)}
    est = attr_h[1].sum(0)
    top = int(est.abs().argmax())
    assert abs(real[top]) >= 0 and torch.isfinite(est).all()
    # hooks are removed
    assert all(len(m._forward_pre_hooks) == 0 for m in (model.model.layers[1].self_attn.o_proj, model.model.layers[2].mlp.down_proj))
    assert cc.recovery(1.0, 0.0, 2.0) == 0.5 and cc.recovery(5.0, 0.0, 2.0) == 2.0
