# Person B -- Lack of Knowledge

**Live entry points (in order):**
1. `python person_B_lack_of_knowledge/screen_templates.py` -- GPU, bf16 model. Measures every NEC
   template on matched fabricated/real entities and writes `template_screen_results.json`.
2. `python person_B_lack_of_knowledge/rebuild_lack_of_knowledge_whitelisted.py` -- tokenizer only.
   Reads the whitelist (gap >= 0.15) from that JSON and writes `data/lack_of_knowledge/{prompts,controls}.jsonl`
   plus `template_quota_report.json`.

`nec_templates.py` (stdlib only) holds the 78 NEC templates, the template classifier, and
`fetch_unknownbench()` which clones UnknownBench and checks out the pinned commit
`7283e4218b9146275d3069306927c3289fad576a`. Run `python person_B_lack_of_knowledge/nec_templates.py --fetch`
to inspect classification without a model.

## What the category measures
UnknownBench NEC questions about fabricated entities ("What is the capital city of Bupseophin?")
with `The answer is` prefilled into the assistant turn. Only templates that empirically separate
real from fabricated entities under this prefill are used (most NEC templates do not -- they were
written for free-text evaluation).

## Controls (`controls.jsonl`, `is_control: true`)
The answerable NEC items (real entities) from the SAME templates with IDENTICAL per-template
quotas (proportional to `min(unanswerable_pool, answerable_pool)`, same seed), so the two files
differ only in whether the entity exists.

## Extra record fields
`template` (NEC template string), `nec_category` (animals / food / countries / medicines / sports / generic).

## Known limitations
- ~15% of NEC prompts match no template for their category and are dropped (reported by `load_and_classify`).
- Not every whitelisted template fails the same way for a fabricated entity: capital/language/government
  fail via fact-retrieval collapse; "interesting behaviors", "interacts with medications" fail closer to
  entity-recognition. Both count as "does not know" here.
- The committed whitelist was derived from an earlier screening run whose per-template table was not
  saved; `template_screen_results.json` must be regenerated in bf16 before rebuilding.
