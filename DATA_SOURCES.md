# Data Sources

| Category | Dataset | Source | Notes |
|---|---|---|---|
| Ambiguity | AmbigQA / AmbigNQ | github.com/shmsw25/AmbigQA, nlp.cs.washington.edu/ambigqa | Filter to items with 2+ disambiguated answers |
| Lack of knowledge | UnknownBench (NEC subset) | github.com/genglinliu/UnknownBench | Use Non-Existent Concepts subset; matched answerable/unanswerable pairs already built in |
| Contradictory context | Knowledge-probing framework (Tighidet et al., 2024) | github.com/Zineddine-Tighidet/knowledge-probing-framework | Built on ParaRel; run their generation scripts to produce contradiction triples |

## Common schema (all three categories convert into this before use)

Save each category's processed data as `data/<category>/prompts.jsonl`, one JSON object per line:

```json
{
  "prompt_id": "amb_0001",
  "category": "ambiguity",
  "raw_prompt": "Tomorrow we should go to the",
  "chat_formatted_prompt": "<Llama-3.1 chat template applied>",
  "source_dataset": "AmbigQA",
  "split": "working"
}
```

- `split` is either `"working"` or `"held_out"` (70/30 split, set once and never changed).
- Subsample each category to ~100-150 working items before moving to Phase 2/3/4 — you don't need
  the full source dataset for a first pass.
- Hand-check ~10-15 items per category before trusting the category label.
