# archive/pipelines -- superseded data-construction scripts

Kept for provenance only. None of these are entry points; each has a single live replacement.

| Archived file | Was | Why archived | Live replacement |
|---|---|---|---|
| `A_build_dataset.py` | naive first-pass AmbigQA filter | no contamination filters, no prefill, old schema | `person_A_ambiguity/preprocess_ambiguity.py` |
| `A_preprocess_ambiguity_v1.py` | Stage 1-2 filters + manual-review Stage 3 via `approval_tracker.jsonl` | the approval tracker was never committed; its time-reference filters (`has_resolvable_time_ref`, `flag_unscoped_present_tense_split`) are now ported into the live script as Stage 2t | same |
| `A_old_preprocess_ambiguity_v2.py` | first deterministic rewrite | baked `The answer is` into the user turn (turn-boundary bug), no controls | same |
| `B_build_dataset.py` | placeholder NEC loader | guessed field names / paths, no prefill | `person_B_lack_of_knowledge/rebuild_lack_of_knowledge_whitelisted.py` |
| `B_old_preprocess_lack_of_knowledge.py` | sentence-starter factoid filter | filter screens phrasing, not whether the template separates known/unknown; `clone_repo` moved to `nec_templates.fetch_unknownbench` (now pinned) | `screen_templates.py` + `rebuild_lack_of_knowledge_whitelisted.py` |
| `B_preprocess_lack_of_knowledge_v2.py` | position-fixed version of the above | still used the starter filter; running it CLOBBERS the whitelisted data | same |
| `C_build_dataset.py` | first CounterFact builder | knows-fact check ignored token identity; no prefill; unpinned | `person_C_contradictory_context/preprocess_contradictory_context.py` |
| `C_old_preprocess_contradictory_context.py` | CPU-only builder with ParaRel overrides | no knows-fact filter, no prefill; `build_template_overrides`/`get_field` moved into the live script (now per-template, with the comma-fragment rule and a curated bad-tail list) | same |
| `C_diagnose_templates.py` | one-off template diagnostic | archived as-is (note: its first line is a stray comment, harmless); the live script prints the same information with `--verbose` | same |

The committed `data/*/*.jsonl` files were produced by earlier versions of these pipelines and
must be regenerated with the live scripts (see `PHASE3_GUIDE.md`).
