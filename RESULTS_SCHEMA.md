# Results CSV Schema

Every person's Phase 3/4 results go in `person_X_<category>/results/results.csv` using this
exact schema — this is what makes the Week 9 merge a simple pandas concat instead of a mess.

| Column | Type | Description |
|---|---|---|
| `neuron_id` | string | e.g. `"L20_N1083"` (layer 20, neuron 1083) |
| `layer` | int | Layer index |
| `neuron_idx` | int | Neuron index within layer |
| `category` | string | `"ambiguity"` / `"lack_of_knowledge"` / `"contradictory_context"` |
| `prompt_id` | string | Matches `prompt_id` in the data file |
| `orig_entropy` | float | Entropy before ablation |
| `ablated_entropy` | float | Entropy after mean-ablation |
| `entropy_shift` | float | `ablated_entropy - orig_entropy` |
| `orig_top1_prob` | float | Top-1 token probability before ablation |
| `ablated_top1_prob` | float | Top-1 token probability after ablation |
| `direct_effect_score` | float | From logit lens (Phase 3), magnitude of direct effect on top predicted token |
| `split` | string | `"working"` or `"held_out"` |

Example row:
```
L20_N1083,20,1083,ambiguity,amb_0001,2.31,1.98,-0.33,0.41,0.52,0.02,working
```

Do not rename columns or change types without telling the team — the merge script in
`results/merge_and_analyze.py` depends on this exact schema.
