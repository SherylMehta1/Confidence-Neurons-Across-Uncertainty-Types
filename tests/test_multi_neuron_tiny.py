"""Smoke test for scripts/multi_neuron_behavioral.py on a tiny random Llama (CPU)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

from test_shared_tiny import make_tiny_model, make_tiny_tokenizer  # noqa: E402
import multi_neuron_behavioral as mb  # noqa: E402


def test_register_set_clamps_all_members_last_position_only():
    model, tok = make_tiny_model(), make_tiny_tokenizer()
    neurons = [(1, 3), (1, 7), (2, 5)]
    values = {(1, 3): 9.0, (1, 7): -9.0, (2, 5): 4.0}
    handles = mb.register_set(model, neurons, values)
    seen = {}
    spies = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
        (lambda L: (lambda m, a: seen.__setitem__(L, a[0].detach().clone())))(l)) for l in (1, 2)]
    enc = tok([f"{tok.bos_token} a b c"], return_tensors="pt", add_special_tokens=False)
    enc.pop("token_type_ids", None)
    with torch.no_grad():
        model(**enc)
    for h in handles + spies:
        h.remove()
    assert torch.allclose(seen[1][0, -1, 3], torch.tensor(9.0))
    assert torch.allclose(seen[1][0, -1, 7], torch.tensor(-9.0))
    assert torch.allclose(seen[2][0, -1, 5], torch.tensor(4.0))
    assert not torch.allclose(seen[1][0, 0, 3], torch.tensor(9.0))
    assert len(model.model.layers[1].mlp.down_proj._forward_pre_hooks) == 0


def test_parse_sets_and_confidence_followup():
    model, tok = make_tiny_model(), make_tiny_tokenizer()
    assert mb.parse_sets(["key=L31_N11541,L30_N1457"]) == {"key": [(31, 11541), (30, 1457)]}
    outs, nums = mb.confidence_followup(model, tok, [f"{tok.bos_token} a the answer is"], [" b c"], max_new_tokens=3, batch_size=2)
    assert len(outs) == 1 and len(nums) == 1
    assert mb.NUM_RE.search("I am about 85 percent sure").group(1) == "85"
    assert mb.NUM_RE.search("100").group(1) == "100"


if __name__ == "__main__":
    test_register_set_clamps_all_members_last_position_only(); test_parse_sets_and_confidence_followup(); print("ok")
