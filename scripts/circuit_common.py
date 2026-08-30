"""
Shared helpers for the circuit pipeline (scripts/circuit_*.py): twin loading and token alignment, the hedge
log-odds readout, per-component hooks (residual stream, attention-head outputs, MLP outputs, MLP neurons),
activation patching between twins, and attribution patching (gradient x activation difference).

Conventions
  - A twin pair is (uncertain record, control record) from twin_patching.make_pairs (twin_id / template / index).
  - Tokens of a twin pair are aligned as  prefix | entity span | suffix : prefix and suffix are the longest common
    token prefix / suffix of the two prompts; the entity span is whatever differs. Positions in the suffix (which
    contains the question tail and the prefill) are aligned right-to-left; the entity span is aligned right-to-left
    as well (its last token is the "entity's last token"), extra tokens on the longer side stay unpatched.
  - Head output = the slice of the o_proj INPUT belonging to that head (before the output projection), so patching a
    head replaces exactly that head's contribution. MLP neuron = a coordinate of the down_proj INPUT.
  - Metric = hedge-vs-answer log-odds at the last position (direction_bridge.HEDGE_TOKENS / ANSWER_TOKENS), and
    recovery = (patched - target) / (source - target) for a pair, clipped to [-1, 2].
All functions work on a HF Llama / Qwen2 model in eval mode; gradients are only enabled inside attribution().
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import contextlib
import json

import numpy as np
import torch

from _common import REPO_ROOT, load_category
from direction_bridge import ANSWER_TOKENS, HEDGE_TOKENS, token_ids
from shared.model_utils import tokenize_prompt
from twin_patching import make_pairs


# ----------------------------------------------------------------------------- pairs and alignment
def load_pairs(category, limit=None, split=None):
    prompts, ctrls = load_category(category)
    pairs, how = make_pairs(prompts, ctrls)
    if split:
        pairs = [(u, c) for u, c in pairs if u.get("split") == split]
    if limit:
        pairs = pairs[:limit]
    return pairs, how


def align(ids_u, ids_c):
    """Return dict with prefix length, entity spans (start, end) for both, and suffix length."""
    n = min(len(ids_u), len(ids_c))
    suf = 0  # the shared tail (question tail + prefill) is matched first: it is what every readout sits on
    while suf < n and ids_u[-1 - suf] == ids_c[-1 - suf]:
        suf += 1
    pre = 0
    while pre < n - suf and ids_u[pre] == ids_c[pre]:
        pre += 1
    return dict(prefix=pre, suffix=suf, ent_u=(pre, len(ids_u) - suf), ent_c=(pre, len(ids_c) - suf), len_u=len(ids_u), len_c=len(ids_c))


def position_map(al, group):
    """List of (target_pos, source_pos) pairs for a position group, target = uncertain prompt, source = control.
    groups: 'entity' (right-aligned within the entity spans), 'suffix' (every suffix token, right-aligned),
    'last' (final token only), 'prefix' (every shared prefix token)."""
    if group == "last":
        return [(al["len_u"] - 1, al["len_c"] - 1)]
    if group == "prefix":
        return [(i, i) for i in range(al["prefix"])]
    if group == "suffix":
        return [(al["len_u"] - 1 - k, al["len_c"] - 1 - k) for k in range(al["suffix"])]
    if group == "entity":
        (su, eu), (sc, ec) = al["ent_u"], al["ent_c"]
        k = min(eu - su, ec - sc)
        return [(eu - 1 - j, ec - 1 - j) for j in range(k)]
    raise ValueError(group)


def reverse_map(pmap):
    return [(s, t) for t, s in pmap]


# ----------------------------------------------------------------------------- readout
class Readout:
    def __init__(self, tokenizer):
        self.hedge_ids = token_ids(tokenizer, HEDGE_TOKENS)
        self.answer_ids = token_ids(tokenizer, ANSWER_TOKENS)

    def logodds(self, logits_last):
        lp = torch.log_softmax(logits_last.float(), -1)
        return torch.logsumexp(lp[..., self.hedge_ids], -1) - torch.logsumexp(lp[..., self.answer_ids], -1)

    def entropy(self, logits_last):
        lp = torch.log_softmax(logits_last.float(), -1)
        return -(torch.xlogy(lp.exp(), lp.exp())).sum(-1)


def recovery(patched, target, source, clip=(-1.0, 2.0)):
    den = source - target
    if abs(den) < 1e-6:
        return float("nan")
    return float(np.clip((patched - target) / den, *clip))


# ----------------------------------------------------------------------------- hooks
def n_heads(model):
    return model.config.num_attention_heads


def head_dim(model):
    return getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads


def component_module(model, layer, kind):
    blk = model.model.layers[layer]
    return {"resid": blk, "heads": blk.self_attn.o_proj, "mlp": blk.mlp, "neurons": blk.mlp.down_proj}[kind]


class Capture:
    """Capture activations of a component kind at given layers. For 'resid' and 'mlp' the module OUTPUT is stored;
    for 'heads' and 'neurons' the module INPUT (o_proj input / down_proj input) is stored. Tensors are detached
    clones on the model device unless keep_graph=True."""

    def __init__(self, model, layers, kind, keep_graph=False):
        self.model, self.layers, self.kind, self.keep_graph = model, layers, kind, keep_graph
        self.acts, self.handles = {}, []

    def __enter__(self):
        for l in self.layers:
            mod = component_module(self.model, l, self.kind)
            if self.kind in ("heads", "neurons"):
                self.handles.append(mod.register_forward_pre_hook(self._pre(l), with_kwargs=False))
            else:
                self.handles.append(mod.register_forward_hook(self._post(l)))
        return self

    def _pre(self, l):
        def hook(module, args):
            x = args[0]
            if not self.keep_graph:
                self.acts[l] = x.detach().clone(); return None
            # parameters do not require grad, so the first hooked tensor is a leaf we create here; later hooked
            # tensors are already in the graph (downstream of that leaf) and only need retain_grad
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            x.retain_grad(); self.acts[l] = x
            return (x,) + tuple(args[1:])
        return hook

    def _post(self, l):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if not self.keep_graph:
                self.acts[l] = h.detach().clone(); return None
            if not h.requires_grad:
                h = h.detach().requires_grad_(True)
            h.retain_grad(); self.acts[l] = h
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


class Patch:
    """Overwrite parts of a component's activation during a forward pass.
    edits: list of (layer, kind, position_pairs, source_act, index) where
      kind 'resid'/'mlp' -> replace output[:, tgt_pos, :] with source_act[:, src_pos, :]
      kind 'heads'       -> replace the head slice (index = head id) of the o_proj input at tgt positions
      kind 'neurons'     -> replace the neuron coordinate (index = neuron id) of the down_proj input at tgt positions
    A position pair (t, s) maps target position t to source position s; source_act is the captured tensor for the
    source run (batch 1)."""

    def __init__(self, model, edits):
        self.model, self.edits, self.handles = model, edits, []
        self.hd = head_dim(model)

    def __enter__(self):
        by_mod = {}
        for e in self.edits:
            by_mod.setdefault((e[0], e[1]), []).append(e)
        for (l, kind), es in by_mod.items():
            mod = component_module(self.model, l, kind)
            if kind in ("heads", "neurons"):
                self.handles.append(mod.register_forward_pre_hook(self._pre(es, kind)))
            else:
                self.handles.append(mod.register_forward_hook(self._post(es)))
        return self

    def _apply(self, x, es, kind):
        x = x.clone()
        for (_, _, pmap, src, idx) in es:
            for t, s in pmap:
                if kind == "heads":
                    a, b = idx * self.hd, (idx + 1) * self.hd
                    x[:, t, a:b] = src[:, s, a:b].to(x.dtype)
                elif kind == "neurons":
                    x[:, t, idx] = src[:, s, idx].to(x.dtype)
                else:
                    x[:, t, :] = src[:, s, :].to(x.dtype)
        return x

    def _pre(self, es, kind):
        def hook(module, args):
            return (self._apply(args[0], es, kind),) + tuple(args[1:])
        return hook

    def _post(self, es):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (self._apply(output[0], es, "resid"),) + tuple(output[1:])
            return self._apply(output, es, "resid")
        return hook

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# ----------------------------------------------------------------------------- forward helpers
@torch.no_grad()
def forward_last(model, enc):
    return model(**enc, use_cache=False).logits[0, -1]


@torch.no_grad()
def run_capture(model, enc, layers, kind):
    with Capture(model, layers, kind) as cap:
        logits = model(**enc, use_cache=False).logits[0, -1]
    return logits, cap.acts


@torch.no_grad()
def run_patched(model, enc, edits):
    with Patch(model, edits):
        return model(**enc, use_cache=False).logits[0, -1]


def attribution(model, enc_target, layers, kind, readout, source_acts, pmaps):
    """Attribution patching: for each layer/component, score = sum over patched positions and features of
    (source - target) * d metric / d activation, evaluated at the target run (first-order estimate of the
    patched-minus-target metric change). Returns {layer: tensor[positions?, features]} aggregated per feature
    group: heads -> [n_heads] (summed over the head's dims and over the mapped positions, also per position),
    neurons -> [d_mlp]; resid/mlp -> scalar per position."""
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.enable_grad():
        # build the graph from the embeddings so every hooked activation requires grad (parameters do not) and the
        # gradient reaches earlier hooked layers through every path, attention branches included
        emb = model.get_input_embeddings()(enc_target["input_ids"]).detach().requires_grad_(True)
        with Capture(model, layers, kind, keep_graph=True) as cap:
            logits = model(inputs_embeds=emb, attention_mask=enc_target.get("attention_mask"), use_cache=False).logits[0, -1]
            metric = readout.logodds(logits)
            metric.backward()
        out = {}
        for l in layers:
            a, g = cap.acts[l], cap.acts[l].grad
            if g is None:
                raise RuntimeError(f"attribution: no gradient reached layer {l} ({kind})")
            diff = torch.zeros_like(a)
            for t, s in pmaps[l] if isinstance(pmaps, dict) else pmaps:
                diff[:, t, :] = source_acts[l][:, s, :].to(a.dtype) - a[:, t, :]
            contrib = (diff.float() * g.float())[0]  # [seq, features]; multiply in fp32, not bf16
            if kind == "heads":
                hd = head_dim(model)
                out[l] = contrib.view(contrib.shape[0], -1, hd).sum(-1).detach().cpu()  # [seq, n_heads]
            elif kind == "neurons":
                out[l] = contrib.detach().cpu()  # [seq, d_mlp]
            else:
                out[l] = contrib.sum(-1).detach().cpu()  # [seq]
    model.zero_grad(set_to_none=True)
    return out


def encode_pair(model, tokenizer, u, c):
    enc_u = tokenize_prompt(tokenizer, u["chat_formatted_prompt"], device=model.device)
    enc_c = tokenize_prompt(tokenizer, c["chat_formatted_prompt"], device=model.device)
    al = align(enc_u["input_ids"][0].tolist(), enc_c["input_ids"][0].tolist())
    assert al["suffix"] >= 1, "twins must share their final token(s) (the prefill); got none in common"
    return enc_u, enc_c, al


def write_json(path, obj):
    path = _Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
