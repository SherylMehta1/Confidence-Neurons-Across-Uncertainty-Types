"""
Tiny-model tests for the shared library. No gated model needed: a seeded
random LlamaForCausalLM (3 layers, hidden 64) plus a minimal word-level
PreTrainedTokenizerFast with a real BOS token.

Run:  python -m pytest tests -q      or      python tests/test_shared_tiny.py
"""

import json
import random
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from shared.model_utils import tokenize_prompt, compute_entropy, get_next_token_probs  # noqa: E402
from shared.detection import (capture_intermediate_activations, detect_candidate_neurons_split_half,  # noqa: E402
                              load_candidate_neurons, split_half_indices, correlate_layer)
from shared.ablation import (RESULT_COLUMNS, mean_ablate_and_get_probs, frozen_norm_ablate_and_get_probs,  # noqa: E402
                             activation_sweep_and_get_probs, get_probs_and_activation)
from shared.prompt_format import seeded_shuffle  # noqa: E402
from shared.run_ablation_pipeline import run_category  # noqa: E402

warnings.simplefilter("ignore", DeprecationWarning)

VOCAB = 256
N_WORDS = 200


def make_tiny_model(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                      bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = LlamaForCausalLM(cfg).eval()
    model.cn_precision = "fp32"
    model.cn_model_id = "tiny-llama-test"
    return model


def make_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"[PAD]": 0, "<s>": 1, "</s>": 2, "[UNK]": 3}
    for i in range(N_WORDS):
        vocab[f"w{i}"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                   unk_token="[UNK]", pad_token="[PAD]")


_CACHE = {}


def fixtures():
    if "model" not in _CACHE:
        _CACHE["model"] = make_tiny_model()
        _CACHE["tok"] = make_tiny_tokenizer()
    return _CACHE["model"], _CACHE["tok"]


def rand_prompt(rng, n=None):
    n = n or rng.randint(4, 12)
    return " ".join(f"w{rng.randrange(N_WORDS)}" for _ in range(n))


def no_hooks(module):
    return len(module._forward_pre_hooks) == 0 and len(module._forward_hooks) == 0


def record_input(store, key):
    """Pre-hook that records the module's first input (returns None so the
    forward is untouched)."""
    def hook(module, args):
        store[key] = args[0].detach().clone()
    return hook


# (a) -----------------------------------------------------------------------
def test_tokenize_prompt_adds_bos_once():
    _, tok = fixtures()
    raw = "w1 w2 w3"
    ids = tokenize_prompt(tok, raw)["input_ids"][0].tolist()
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1 and len(ids) == 4
    ids2 = tokenize_prompt(tok, tok.bos_token + " " + raw)["input_ids"][0].tolist()
    assert ids2 == ids, (ids2, ids)
    assert tokenize_prompt(tok, raw)["attention_mask"].shape == (1, 4)


# (b) -----------------------------------------------------------------------
def test_capture_matches_manual_forward():
    model, tok = fixtures()
    prompt = "w5 w6 w7 w8"
    cap = capture_intermediate_activations(model, tok, prompt, [0, 2])
    assert cap[2].shape == (model.config.intermediate_size,)
    # manual: run the model with our own hook recording the full down_proj input
    box = {}
    h = model.model.layers[2].mlp.down_proj.register_forward_pre_hook(record_input(box, "x"))
    with torch.no_grad():
        model(**tokenize_prompt(tok, prompt), use_cache=False)
    h.remove()
    assert np.allclose(cap[2], box["x"][0, -1].numpy(), atol=1e-6)
    assert no_hooks(model.model.layers[2].mlp.down_proj)


# (c) -----------------------------------------------------------------------
def test_mean_ablation_changes_only_last_position_and_hooks_cleanup():
    model, tok = fixtures()
    prompt = "w9 w10 w11 w12 w13"
    layer, neuron = 1, 7
    dp = model.model.layers[layer].mlp.down_proj
    seen = {}
    # A second capture hook records what down_proj finally receives. It is
    # registered BEFORE the ablation pre-hook, so use with_kwargs-free forward
    # hook semantics: forward hooks see the (possibly pre-hook-modified) args.
    def see(module, args, out):
        seen["in"] = args[0].detach().clone()
    h = dp.register_forward_hook(see)
    try:
        clean = capture_intermediate_activations(model, tok, prompt, [layer])
        seen.pop("in")
        probs = mean_ablate_and_get_probs(model, tok, prompt, layer, neuron, 3.0)
        x = seen["in"][0]
        assert abs(x[-1, neuron].item() - 3.0) < 1e-6
        # full check: every position except last, and every neuron except `neuron` at last, equals clean
        full_clean = {}
        h2 = dp.register_forward_pre_hook(record_input(full_clean, "x"))
        with torch.no_grad():
            model(**tokenize_prompt(tok, prompt), use_cache=False)
        h2.remove()
        fc = full_clean["x"][0]
        assert torch.allclose(x[:-1], fc[:-1], atol=1e-6)
        mask = torch.ones_like(fc[-1], dtype=torch.bool); mask[neuron] = False
        assert torch.allclose(x[-1][mask], fc[-1][mask], atol=1e-6)
        assert abs(float(probs.sum()) - 1.0) < 1e-5
    finally:
        h.remove()
    assert no_hooks(dp)

    # exception inside forward -> hooks still removed
    boom = model.model.layers[2].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        try:
            mean_ablate_and_get_probs(model, tok, prompt, layer, neuron, 0.0)
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
        try:
            capture_intermediate_activations(model, tok, prompt, [layer])
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
    finally:
        boom.remove()
    assert no_hooks(dp) and no_hooks(model.model.layers[2].mlp.down_proj)


# (d) -----------------------------------------------------------------------
def test_frozen_norm():
    model, tok = fixtures()
    prompt = "w20 w21 w22 w23 w24 w25"
    layer, neuron = 2, 11
    norm = model.model.norm
    clean = get_next_token_probs(model, tok, prompt)
    act = capture_intermediate_activations(model, tok, prompt, [layer])[layer][neuron]
    frozen_same = frozen_norm_ablate_and_get_probs(model, tok, prompt, layer, neuron, float(act))
    assert torch.allclose(frozen_same, clean, atol=1e-5), (frozen_same - clean).abs().max()

    val = float(act) + 5.0
    frozen = frozen_norm_ablate_and_get_probs(model, tok, prompt, layer, neuron, val)
    full = mean_ablate_and_get_probs(model, tok, prompt, layer, neuron, val)
    assert not torch.allclose(frozen, full, atol=1e-6)
    assert not torch.allclose(frozen, clean, atol=1e-6)
    assert "forward" not in norm.__dict__ and no_hooks(norm) and no_hooks(model.model.layers[layer].mlp.down_proj)
    # and the model is back to normal
    assert torch.allclose(get_next_token_probs(model, tok, prompt), clean, atol=1e-7)

    # restoration also on exception
    boom = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        try:
            frozen_norm_ablate_and_get_probs(model, tok, prompt, layer, neuron, val)
        except RuntimeError:
            pass
    finally:
        boom.remove()
    assert "forward" not in norm.__dict__ and no_hooks(norm)

    # dose-response sweep: first value equal to clean activation reproduces clean
    sweep = activation_sweep_and_get_probs(model, tok, prompt, layer, neuron, [float(act), val])
    assert len(sweep) == 2 and torch.allclose(sweep[0], clean, atol=1e-6)
    assert torch.allclose(sweep[1], full, atol=1e-6)
    probs, a2 = get_probs_and_activation(model, tok, prompt, layer, neuron)
    assert abs(a2 - float(act)) < 1e-6 and torch.allclose(probs, clean, atol=1e-7)


# (e) -----------------------------------------------------------------------
def test_compute_entropy():
    p = torch.softmax(torch.randn(300), dim=-1)
    ref = float(-(p * torch.log(p)).sum())
    assert abs(compute_entropy(p) - ref) < 1e-6
    onehot = torch.zeros(10); onehot[3] = 1.0
    assert compute_entropy(onehot) == 0.0
    assert compute_entropy(np.array([0.5, 0.5])) - np.log(2) < 1e-6


# (f) -----------------------------------------------------------------------
def test_split_half_detection():
    model, tok = fixtures()
    rng = random.Random(0)
    prompts = [rand_prompt(rng) for _ in range(24)]
    labels = ["a"] * 9 + ["b"] * 15
    cands, dist = detect_candidate_neurons_split_half(
        model, tok, prompts, layer_range=[1, 2], top_k_per_half=30, top_k_final=None,
        seed=42, min_abs_corr=0.3, stratify_by=labels, verbose=False)
    prov = dist["provenance"]
    for key in ("model_id", "precision", "dtype", "transformers_version", "torch_version", "seed",
                "layer_range", "top_k_per_half", "top_k_final", "min_abs_corr", "baseline_prompt_sha256",
                "n_baseline_prompts", "git_head_sha", "timestamp", "dropped_zero_variance"):
        assert key in prov, key
    assert prov["n_half_a"] + prov["n_half_b"] == 24
    assert prov["stratum_counts"]["half_a"]["a"] in (4, 5) and prov["stratum_counts"]["half_b"]["b"] in (7, 8)
    for c in cands:
        assert abs(c["detection_correlation_half_a"]) >= 0.3 and abs(c["detection_correlation_half_b"]) >= 0.3
        assert c["layer"] in (1, 2)
    # disjoint halves, balanced strata
    a, b = split_half_indices(24, 42, labels)
    assert set(a).isdisjoint(b) and len(a) + len(b) == 24
    assert abs(sum(labels[i] == "a" for i in a) - sum(labels[i] == "a" for i in b)) <= 1
    # legacy (unstratified) split is the numpy permutation used before
    a0, b0 = split_half_indices(24, 42)
    r = np.random.default_rng(42); idx = np.arange(24); r.shuffle(idx)
    assert a0 == idx[:12].tolist() and b0 == idx[12:].tolist()
    # vectorized correlation == np.corrcoef, with zero-variance columns dropped
    X = np.random.default_rng(1).normal(size=(20, 5)); X[:, 2] = 1.0
    e = np.random.default_rng(2).normal(size=20)
    r_vec, n_drop = correlate_layer(X, e)
    assert n_drop == 1 and np.isnan(r_vec[2])
    for j in (0, 1, 3, 4):
        assert abs(r_vec[j] - np.corrcoef(X[:, j], e)[0, 1]) < 1e-10
    # min_abs_corr actually filters: an impossible threshold yields nothing
    cands2, _ = detect_candidate_neurons_split_half(
        model, tok, prompts, layer_range=[1], top_k_per_half=30, min_abs_corr=1.01, verbose=False)
    assert cands2 == []


# (g) -----------------------------------------------------------------------
def test_load_candidate_neurons_formats():
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "raw.json"; p1.write_text(json.dumps([[30, 1457, 0.78], [29, 11909, -0.6]]))
        p2 = Path(d) / "dicts.json"; p2.write_text(json.dumps([{"neuron_id": "L30_N1457", "layer": 30, "neuron_idx": 1457}]))
        p3 = Path(d) / "wrapped.json"; p3.write_text(json.dumps(
            {"provenance": {"seed": 42}, "candidates": [{"layer": 31, "neuron_idx": 2477}]}))
        p4 = Path(d) / "meta.json"; p4.write_text(json.dumps(
            {"metadata": {}, "neurons": [[31, 2477]]}))
        for p in (p1, p2, p3, p4):
            cands = load_candidate_neurons(p)
            assert all({"neuron_id", "layer", "neuron_idx"} <= set(c) for c in cands)
        assert load_candidate_neurons(p1)[1] == {"neuron_id": "L29_N11909", "layer": 29, "neuron_idx": 11909,
                                                 "detection_correlation": -0.6}
        assert load_candidate_neurons(p4)[0]["neuron_id"] == "L31_N2477"
    # the repo's real file
    real = REPO / "candidate_neurons.json"
    if real.exists():
        assert len(load_candidate_neurons(real)) >= 1


# (h) -----------------------------------------------------------------------
def test_run_category_schema_resume_overwrite():
    import pandas as pd
    model, tok = fixtures()
    rng = random.Random(3)

    def recs(prefix, n, is_control):
        out = []
        for i in range(n):
            out.append({"prompt_id": f"{prefix}_{i:04d}", "category": "tiny", "raw_prompt": "x",
                        "chat_formatted_prompt": rand_prompt(rng), "source_dataset": "synthetic",
                        "split": "working" if i < n * 0.7 else "held_out", "is_control": is_control})
        return out

    prompts, controls = recs("t", 6, False), recs("t_ctrl", 4, True)
    cands = [{"neuron_id": "L1_N3", "layer": 1, "neuron_idx": 3}, {"neuron_id": "L2_N9", "layer": 2, "neuron_idx": 9}]
    with tempfile.TemporaryDirectory() as d:
        df = run_category(model, tok, cands, prompts, controls, "tiny", out_dir=d, verbose=False)
        assert list(df.columns) == RESULT_COLUMNS
        assert len(df) == 2 * (6 + 4)
        assert set(df["split"]) == {"working", "held_out"} and set(df["is_control"]) == {True, False}
        assert set(df["mean_source"]) == {"pooled_controls"} and set(df["precision"]) == {"fp32"}
        assert df.groupby("neuron_id")["mean_val"].nunique().max() == 1
        assert (Path(d) / "results_tiny.provenance.json").exists() and (Path(d) / "ablation_means.json").exists()
        prov = json.loads((Path(d) / "results_tiny.provenance.json").read_text())
        n_work = sum(r["split"] == "working" for r in prompts + controls)
        assert prov["mean_source"] == "pooled_controls" and prov["n_baseline_prompts"] == n_work
        # orig_activation is the real activation
        row = df[(df.neuron_id == "L1_N3") & (df.prompt_id == "t_0000")].iloc[0]
        act = capture_intermediate_activations(model, tok, prompts[0]["chat_formatted_prompt"], [1])[1][3]
        assert abs(row.orig_activation - act) < 1e-6

        # resume: add a neuron, previous ones are not re-run / duplicated
        cands3 = cands + [{"neuron_id": "L0_N1", "layer": 0, "neuron_idx": 1}]
        df2 = run_category(model, tok, cands3, prompts, controls, "tiny", out_dir=d, verbose=False)
        assert len(df2) == 3 * 10 and df2.duplicated(["neuron_id", "prompt_id"]).sum() == 0
        # refuse to overwrite
        try:
            run_category(model, tok, cands3, prompts, controls, "tiny", out_dir=d, resume=False, verbose=False)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        # explicit overwrite restarts from scratch
        df3 = run_category(model, tok, cands, prompts, controls, "tiny", out_dir=d, overwrite=True, verbose=False)
        assert len(df3) == 2 * 10
        # general baseline policy + category_working
        df4 = run_category(model, tok, cands, prompts, controls, "tiny", out_dir=d, overwrite=True, verbose=False,
                           baseline_prompts_for_mean=[rand_prompt(rng) for _ in range(5)])
        assert set(df4["mean_source"]) == {"general_baseline"}
        df5 = run_category(model, tok, cands, prompts, controls, "tiny", out_dir=d, overwrite=True, verbose=False,
                           mean_source="category_working")
        assert set(df5["mean_source"]) == {"category_working"}
        pd.read_csv(Path(d) / "results_tiny.csv")  # parseable


# (i) -----------------------------------------------------------------------
def test_seeded_shuffle_matches_legacy_global_seed():
    items = [f"p{i}" for i in range(173)]
    legacy = list(items); random.seed(42); random.shuffle(legacy)
    assert seeded_shuffle(items, 42) == legacy
    random.seed(7); legacy7 = list(items); random.shuffle(legacy7)
    assert seeded_shuffle(items, 7) == legacy7


# stolfo criteria smoke test on the tiny model ----------------------------------
def test_stolfo_criteria_smoke():
    sys.path.insert(0, str(REPO / "analysis"))
    from analysis.stolfo_criteria import analyze
    model, _ = fixtures()
    cands = [{"neuron_id": "L1_N3", "layer": 1, "neuron_idx": 3}, {"neuron_id": "L2_N9", "layer": 2, "neuron_idx": 9}]
    with tempfile.TemporaryDirectory() as d:
        df = analyze(model, cands, out_csv=Path(d) / "s.csv", n_random=40, n_matched=5, seed=1,
                     null_ks=(4, 16), verbose=False)
        assert {"w_norm", "logit_var", "nullfrac_k4", "nullfrac_k16", "nullfrac_bottom10pct",
                "w_norm_pctile_random", "nullfrac_k16_pctile_matched"} <= set(df.columns)
        assert (df.kind == "candidate").sum() == 2 and (df.kind == "random").sum() == 40
        assert (Path(d) / "s.provenance.json").exists() and (Path(d) / "s_summary.txt").exists()


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback; traceback.print_exc()
            print(f"FAIL {n}: {e}")
    print(f"{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)
