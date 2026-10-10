# Round-3 experiments (UniReps 2026 extended abstract)

Each experiment had a criterion note written and hashed before it ran. The notes live in the paper repo (`Sher/paper/PREREG/`), with their SHA-256 prefixes recorded in `PREREG/HASHES_round3.txt`. Read-offs come from `python scripts/round3_analysis.py <exp>`, which writes `<exp>_summary.txt`.

| Exp | Question | Criterion (sha) | Script | Outputs | Verdict |
|---|---|---|---|---|---|
| E1 | Do knowledge labels from an independent prompt keep the abstention result? | e1_independent_labels.md (af50f53f) | e1_relabel.py | e1_llama.jsonl, e1_summary.txt | SURVIVES; entropy NOT CIRCULAR |
| E2 | Does name surprisal stand in for familiarity? | e2_name_surprisal.md (2b69e06c) | e2_surprisal.py | e2_surprisal.jsonl, e2_summary.txt | NOT SUPPORTED |
| E3 | Is the base-model lens popularity effect in the hedge term? | e3_lens_split.md (b2b916ce) | e3_lens_split.py | e3/<model>/lens.csv, e3_window_effects.csv, e3_summary.txt | HEDGE-SIDE (4/4) |
| E4 | Steering with a held-out dose and an abstention-orthogonal direction | e4_steering_v3.md (18c74883); amendment e4_amendment_third_judge.md (7bed3826) | e4_steer.py | e4/, e4_summary.txt | D1 NOT MET; D2 CARRIED BY THE ABSTENTION COMPONENT |
| E5 | Third judge (Llama-3.3-70B) | e5_third_judge.md (f35aa4d9) | hedge_measure_v2.py judge | ../hedge_measure_v2/judge_llama70b_*.jsonl, e5_summary.* | Llama FOLLOWS POPULARITY; Gemma NEITHER (near floor); Qwen, Mistral UNMEASURED |
| E6 | Entity-level knowledge as a covariate | e6_entity_knowledge.md (937b46e3) | e6_fetch_wikidata.py, e6_entity.py | e6_*.jsonl, e6_summary.txt | SURVIVES ENTITY KNOWLEDGE |
| E7 | Semantic relabelling of unknown facts; strict declines | e7_label_and_decline_audit.md (1498be91) | e7_audit.py | e7_answers/grades/declines.jsonl, e7_summary.txt | SURVIVES SEMANTIC RELABELLING; STRICT-SURVIVES |
| E8 | Steering with an abstention direction fitted within the obscure arm | e8_steering_within_level.md (05474954) | e8_steer.py | e8/, e8_summary.txt | CARRIED BY ABSTENTION |
| E9 | Prompt that permits "I don't know" (Llama, Gemma) | e9_permissive_prompt.md (f6459163) | e9_permissive.py | e9/, e9_summary.txt | Llama and Gemma FOLLOW POPULARITY |
| E10 | Instruct-model lens vs base | e10_instruct_lens.md (6af179ce) | e3_lens_split.py | e3/<model>-it/, e10_summary.* | NO CHANGE SHOWN (instruct leans towards knowledge) |

Other folders:
- `e8_INVALID_swapped_kstar/`: the first E8 run, launched with the two models' doses swapped (Llama at 4x, Gemma at 1x). It is kept for the record and not used; see `Sher/paper/LAB_LOG.md`.
- `_boxlogs/box<N>/`: stdout of the remote GPU runs (molab boxes 1-6, 2026-10-10).

Exploratory, not in the paper: OLMo-2-7B-Instruct through the E5 pipeline (`../../data/familiarity_v3_olmo2_7b/`, `../hedge_measure_v2/*olmo2_7b*`; read-off in `Sher/paper/analysis/explore_olmo.txt`).
