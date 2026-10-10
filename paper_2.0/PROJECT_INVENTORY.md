# Project inventory: Confidence-Neurons-Across-Uncertainty-Types

Generated 2026-10-07 by inventory_project.py. Everything below is read from disk and git; nothing is interpreted.

## Summary

- Files scanned: 1447 (skipped dirs: .agents, .claude, .git, __pycache__)
- Git: 118 commits, 2026-08-06 to 2026-10-04, 4 human author(s) after merging identities, 60 untracked paths, 3 modified
- Results folders: 58, provenance sidecars: 258
- Key documents found: 27, results folders no document mentions: 16, documents mentioned but absent: 11

## Git

- Commits on the current branch: 118 (all branches: 118)
- First commit: 2026-08-06; last commit: 2026-10-04
- Commits with an AI-assistant co-author trailer: 63
- Branches: l0-fixes, origin, origin/l0-fixes, origin/main, origin/uncertainty-circuit-experiments
- Remotes: origin https://github.com/SherylMehta1/Confidence-Neurons-Across-Uncertainty-Types.git

Git identities that share an email, a name or a name prefix are merged into one author row.

| Author | Commits | First | Last | Other names used | Emails | Note |
|---|---|---|---|---|---|---|
| SherylMehta1 | 68 | 2026-08-06 | 2026-10-04 | Sheryl Mehta | [email redacted] |  |
| shivam | 31 | 2026-08-22 | 2026-09-02 | shivam dubey | [email redacted], [email redacted] |  |
| shagunchadha | 11 | 2026-08-09 | 2026-08-18 | Shagun Chadha | [email redacted] |  |
| kanikasinghal1612 | 7 | 2026-08-09 | 2026-08-18 | kanika singhal, SinghalKanika | [email redacted] |  |
| your_github_username | 1 | 2026-08-13 | 2026-08-13 |  | your_email@example.com | placeholder git identity (unset user.name or user.email) |

### Commits by date

| Date | Commits | Subjects (first three) |
|---|---|---|
| 2026-08-06 | 1 | initial project scaffold |
| 2026-08-07 | 1 | fix: tool 1-5 sanity check code |
| 2026-08-08 | 2 | fix: split label consistency + import paths for preprocessing scripts; fix toolbox bugs + reformat lack-of-knowledge prompts to cloze-style |
| 2026-08-09 | 7 | Filtering lack of knowledge prompts for next token experiments; Fix broken CounterFact templates using ParaRel relation-id matching; Add preprocessed contradictory-context prompts (84 working / 36 held-o |
| 2026-08-10 | 1 | fix: correct neuron space in ablation to match detection/logit_lens |
| 2026-08-11 | 4 | fix: normalize load_candidate_neurons() to always return dicts, regard; Phase 3: mechanism check on frozen candidate neurons (shared, category; Phase 4 complete for lack_of_knowledge: 4/15 candidates show significa |
| 2026-08-12 | 3 | Add Phase 4 diagnostics for contradictory_context: spread, significanc; Phase 4 complete for ambiguity: 3/15 candidates show significant effec; Merged tests and diagnostics |
| 2026-08-13 | 5 | Phase 1 remediation: fp32 entropy, double-BOS fix, exception-safe hook; Update Phase 4 results and RMSNorm-corrected mechanism check for contr; Update results with fixed Phase 1 pipeline for lack of knowledge |
| 2026-08-16 | 9 | Phase 2 remediation for contradictory_context: unquantized (bf16) abla; Rename pre-audit mechanism check to mechanism_check_shared_old.json, m; Add Phase 2 analysis results for Lack of Knowledge |
| 2026-08-17 | 14 | Fixed B phase 3 rem preprocess file; Fix 2 B phase 3 rem; Screen temp for B phase 3 rem |
| 2026-08-18 | 4 | Person B marimo nb; Person B and C notebooks; Revise README for project clarity and structure |
| 2026-08-22 | 1 | Rebuild pipeline in bf16 with matched controls; audit fixes, scripts,  |
| 2026-08-24 | 7 | Step 1: model-relative uncertainty definition, gated twin datasets, ju; Circuit pipeline and results: the uncertainty circuit in two models; Llama contested circuit outputs and faithfulness files (final sandbox  |
| 2026-08-26 | 12 | Act on external audit: Option-B claim 7 (directional asymmetry, all ce; RESULTS.md: synced abstract and criterion wording to the post-audit fr; Decider experiments: rank-1 direction patching vs the component set; g |
| 2026-08-28 | 1 | Carlini rewrite: one idea (decision vs spread), formula abstract, stor |
| 2026-08-29 | 3 | Tri-audit fixes: direction glossary enforced (remove/inject glosses), ; Team memo: direction glossary, per-claim corrections, and drafts for t; Final read-through fixes: restore eaten sentence, quote typo, audited  |
| 2026-08-30 | 7 | EXPLAINER.md: plain-language walkthrough of the project, all seven cla; bug fix; Merge pull request #1 from SherylMehta1/uncertainty-circuit-experiment |
| 2026-09-02 | 3 | Random-direction injection null, direction paired tests, section 3/4 p; Cold-panel camera-ready round: BH sentence corrected, CI bands on all ; Merge pull request #2 from SherylMehta1/uncertainty-circuit-experiment |
| 2026-09-12 | 2 | L0 fixes: retention, checkpoint/resume, configurable class flags, n-fr; Re-run familiarity twins with L0 fixes applied (data/familiarity_v2) |
| 2026-09-13 | 2 | Dedupe PopQA rows by (relation, subject) before candidate selection; Regenerate familiarity_v2 with the PopQA dedup fix applied |
| 2026-09-14 | 1 | Regenerate familiarity circuit pipeline on the deduped twins (familiar |
| 2026-09-15 | 2 | U/U hedger-vs-confabulator arm + --hedging-category for direction_brid; bridge_familiarity_v3: hedging direction sourced from the U/U arm |
| 2026-09-16 | 4 | L1 dissociation figure for familiarity_v2, with per-pair bootstrap CIs; Pair-level bootstrap CI on cos(familiarity, hedging) for bridge_famili; Within-pair label-permutation null for cos(familiarity, hedging) |
| 2026-09-19 | 7 | L2 step 1+2: leave-one-token-out and causal timing (scripts; not yet r; circuit_causal_timing.py: --positions flag (all / suffix / entity / la; L2 step 1+2 results: LOTO and causal timing on familiarity_v2 |
| 2026-09-25 | 1 | make_numbers_tex.py: generate paper/numbers.tex from committed result  |
| 2026-09-26 | 2 | L2e saturation sweep (no GPU needed) + 13 saturation macros; 53-vs-53 bridge rerun: the wide null is geometric, not a small-sample  |
| 2026-09-27 | 1 | fig_dissociation --suffix, loto_recompute --reference-suffix (for full |
| 2026-09-28 | 1 | 470-pair pool reruns at limit 141: LOTO, causal timing, full-scale fig |
| 2026-10-01 | 1 | PRE-REGISTRATION: U/U hedger-vs-confabulator falsification test |
| 2026-10-02 | 8 | circuit_faithfulness.py: --eval-split, for the U/U exploratory pass; U/U steps A-D + step E power re-check; GO decision recorded BEFORE the; U/U CONFIRMATORY RESULT: passes the locked criterion, but on a knife e |
| 2026-10-04 | 1 | Regeneration audit: the pipeline is clean, but main.tex cites 0 of 239 |

Use --all-commits for every commit line.

### Untracked paths (60)

- data/factorial_v3_gemma_know_given_hi/
- data/factorial_v3_gemma_know_given_lo/
- data/factorial_v3_gemma_pop_given_known/
- data/factorial_v3_gemma_pop_given_unknown/
- data/factorial_v3_gemma_summary.csv
- data/factorial_v3_mistral_know_given_hi/
- data/factorial_v3_mistral_know_given_lo/
- data/factorial_v3_mistral_pop_given_known/
- data/factorial_v3_mistral_pop_given_unknown/
- data/factorial_v3_mistral_summary.csv
- data/factorial_v3_qwen_know_given_hi/
- data/factorial_v3_qwen_know_given_lo/
- data/factorial_v3_qwen_pop_given_known/
- data/factorial_v3_qwen_pop_given_unknown/
- data/factorial_v3_qwen_summary.csv
- data/familiarity_v3/hedge_logodds.jsonl
- data/familiarity_v3/hedge_logodds.provenance.json
- data/familiarity_v3_gemma/
- data/familiarity_v3_mistral/
- data/familiarity_v3_qwen/
- direction_reuse_criterion.md
- direction_reuse_v2_criterion.md
- gemma_mistral_factorial_criterion.md
- hedge_judge_criterion.md
- hedge_measure_v2_criterion.md
- jlens_cascade_criterion.md
- labels_dr/
- nohedge_direction_criterion.md
- paper/figures_singlebos/
- paper/reviews_interpscience.md
- qwen_factorial_criterion.md
- results/OVERVIEW_2026-10-07.md
- results/direction_nohedge/
- results/direction_reuse/
- results/direction_reuse_v2/
- results/factorial_4models/
- results/gemma_factorial/
- results/hedge_judge/
- results/hedge_measure_v2/
- results/jlens_bos_audit/
- results/jlens_cascade/
- results/mistral_factorial/
- results/qwen_factorial/
- results/singlebos/
- results/unireps_checks/
- scripts/add_hedge_logodds.py
- scripts/direction_nohedge.py
- scripts/direction_reuse.py
- scripts/direction_reuse_v2.py
- scripts/hedge_judge.py
- scripts/hedge_judge_analysis.py
- scripts/hedge_measure_v2.py
- scripts/hedge_measure_v2_analysis.py
- scripts/hedge_measure_v2_human_score.py
- scripts/jlens_bos_audit.py
- scripts/jlens_cascade.py
- scripts/jlens_cascade_analysis.py
- scripts/make_fig45_singlebos.py
- scripts/qwen_factorial_analysis.py
- scripts/unireps_confound_checks.py

### Modified or staged paths (3)

- M scripts/jlens_positions.py
- M scripts/jlens_steer.py
- M scripts/jlens_trajectory.py

## Key documents

| Path | Kind | Modified | Size | First heading |
|---|---|---|---|---|
| README.md | readme | 2026-10-05 | 6.4 KB | Uncertainty Has a Circuit, Not a Neuron |
| RESULTS.md | results | 2026-10-05 | 19.6 KB | No confidence neurons on the behavioral arm — and a faithfulness-tested uncertainty circui |
| RESULTS_SCHEMA.md | results | 2026-10-05 | 3.6 KB | Results CSV Schema (v4) |
| direction_reuse_criterion.md | prereg | 2026-10-06 | 4.8 KB | Does instruction tuning reuse the base model's familiarity direction? Criterion, stated be |
| direction_reuse_v2_criterion.md | prereg | 2026-10-07 | 3.2 KB | Direction reuse, version 2 (position-matched, dose-calibrated): criterion, stated before t |
| gemma_mistral_factorial_criterion.md | prereg | 2026-10-06 | 2.5 KB | Gemma-2-9B-it and Mistral-7B-Instruct-v0.3 factorials: read-off criterion, stated before t |
| hedge_judge_criterion.md | prereg | 2026-10-07 | 3.0 KB | Judge audit of the hedge detector across models: criterion, stated before the audit |
| hedge_measure_v2_criterion.md | prereg | 2026-10-07 | 4.0 KB | Hedging measured properly (96-token answers, two judges, three-way labels): criterion, sta |
| jlens_cascade_criterion.md | prereg | 2026-10-06 | 4.8 KB | Jacobian-lens "familiarity before knowledge" study across models: criterion, stated before |
| l5_behavioral_criterion.md | prereg | 2026-10-05 | 3.5 KB | L5 / step 9 behavioral validation: read-off criterion, stated before the run |
| nohedge_direction_criterion.md | prereg | 2026-10-07 | 2.4 KB | Is the familiarity steering effect just a "hedging intent" direction? Criterion, stated be |
| preregistration_commit.md | prereg | 2026-10-05 | 12.3 KB | Pre-Registration Commit: U/U (Hedger vs. Confabulator) Falsification Test |
| qwen_factorial_criterion.md | prereg | 2026-10-06 | 3.6 KB | Qwen2.5-7B-Instruct popularity x knowledge factorial: read-off criterion, stated before th |
| archive/README.md | readme | 2026-10-05 | 4.6 KB | archive/ |
| paper/main.tex | paper source | 2026-10-05 | 25.9 KB |  |
| paper/numbers.tex | numbers, paper source | 2026-10-05 | 22.5 KB |  |
| paper/references.bib | bibliography | 2026-10-05 | 6.8 KB |  |
| person_A_ambiguity/README.md | readme | 2026-10-05 | 2.1 KB | Person A -- Ambiguity |
| person_B_lack_of_knowledge/README.md | readme | 2026-10-05 | 2.2 KB | Person B -- Lack of Knowledge |
| person_C_contradictory_context/LIMITATIONS.md | limitations | 2026-10-05 | 3.3 KB | Contradictory Context — Known Limitation |
| person_C_contradictory_context/README.md | readme | 2026-10-05 | 1.9 KB | Person C -- Contradictory Context |
| archive/notebooks/README.md | readme | 2026-10-05 | 3.0 KB | Archived marimo notebooks (historical run record -- do not run) |
| results/direction_reuse/gemma2b/result.json | results | 2026-10-07 | 975 B |  |
| results/direction_reuse/gemma9b/result.json | results | 2026-10-07 | 977 B |  |
| results/direction_reuse/llama/result.json | results | 2026-10-07 | 1.1 KB |  |
| results/direction_reuse/mistral/result.json | results | 2026-10-07 | 988 B |  |
| results/direction_reuse/qwen/result.json | results | 2026-10-07 | 973 B |  |

## Results inventory

### archive/results/

- files directly in this folder: 4 (newest 2026-10-05)

### person_A_ambiguity/results/

- files directly in this folder: 2 (newest 2026-10-05)

### person_B_lack_of_knowledge/results/

- files directly in this folder: 2 (newest 2026-10-05)

### person_C_contradictory_context/results/

- files directly in this folder: 2 (newest 2026-10-05)

### results/

- files directly in this folder: 139 (newest 2026-10-07)

| Folder | Files | Size | Newest file |
|---|---|---|---|
| results/ablation_bf16_conflict_key | 4 | 160.4 KB | 2026-10-05 |
| results/ablation_bf16_conflict_new | 4 | 846.0 KB | 2026-10-05 |
| results/ablation_bf16_direction_familiarity | 7 | 4.0 MB | 2026-10-05 |
| results/ablation_bf16_direction_familiarity_pooled | 7 | 3.2 MB | 2026-10-05 |
| results/ablation_bf16_entropy_weights | 8 | 4.1 MB | 2026-10-05 |
| results/ablation_bf16_entropy_weights_pooled | 8 | 3.2 MB | 2026-10-05 |
| results/ablation_bf16_familiarity_dir | 4 | 424.1 KB | 2026-10-05 |
| results/ablation_bf16_familiarity_key | 4 | 130.0 KB | 2026-10-05 |
| results/ablation_bf16_familiarity_new | 5 | 684.8 KB | 2026-10-05 |
| results/ablation_bf16_familiarity_v2_dir | 3 | 1.8 MB | 2026-10-05 |
| results/ablation_bf16_familiarity_v2_key | 3 | 563.7 KB | 2026-10-05 |
| results/ablation_bf16_familiarity_v2_new | 3 | 2.9 MB | 2026-10-05 |
| results/ablation_bf16_frequency_weights | 8 | 4.1 MB | 2026-10-05 |
| results/ablation_bf16_keyset_pooled | 8 | 1.8 MB | 2026-10-05 |
| results/ablation_bf16_new | 8 | 5.2 MB | 2026-10-05 |
| results/ablation_bf16_old15 | 8 | 3.3 MB | 2026-10-05 |
| results/ablation_bf16_v3set | 8 | 3.6 MB | 2026-10-05 |
| results/ablation_conf_ctrl50 | 4 | 1.9 MB | 2026-10-05 |
| results/ablation_fam_ctrl100 | 4 | 2.6 MB | 2026-10-05 |
| results/bridge_familiarity | 6 | 14.8 MB | 2026-10-05 |
| results/bridge_familiarity_v2 | 6 | 14.8 MB | 2026-10-05 |
| results/bridge_familiarity_v3 | 12 | 14.8 MB | 2026-10-05 |
| results/bridge_familiarity_v4 | 10 | 108.5 KB | 2026-10-05 |
| results/bridge_familiarity_v5 | 12 | 14.9 MB | 2026-10-05 |
| results/circuit_aleatoric | 4 | 170.3 KB | 2026-10-05 |
| results/circuit_aleatoric_qwen | 12 | 247.1 KB | 2026-10-05 |
| results/circuit_conflict | 33 | 3.2 MB | 2026-10-05 |
| results/circuit_conflict_qwen | 14 | 2.9 MB | 2026-10-05 |
| results/circuit_factorial_v3_know_given_hi | 23 | 5.8 MB | 2026-10-05 |
| results/circuit_factorial_v3_pop_given_unknown | 23 | 6.5 MB | 2026-10-05 |
| results/circuit_familiarity | 35 | 2.9 MB | 2026-10-05 |
| results/circuit_familiarity_v2 | 49 | 6.3 MB | 2026-10-05 |
| results/circuit_familiarity_v3 | 62 | 8.9 MB | 2026-10-05 |
| results/circuit_uu_hedge_confab_v3 | 28 | 4.0 MB | 2026-10-05 |
| results/direction_nohedge | 5 | 155.0 KB | 2026-10-07 |
| results/direction_reuse | 41 | 1.6 MB | 2026-10-07 |
| results/direction_reuse_v2 | 11 | 850.6 KB | 2026-10-07 |
| results/factorial_4models | 7 | 1.7 MB | 2026-10-07 |
| results/frequency_causal_v1_raw_readout | 4 | 2.6 MB | 2026-10-05 |
| results/gemma_factorial | 4 | 553.2 KB | 2026-10-06 |
| results/hedge_judge | 7 | 502.8 KB | 2026-10-07 |
| results/hedge_measure_v2 | 22 | 2.2 MB | 2026-10-07 |
| results/jlens_bos_audit | 3 | 1.4 MB | 2026-10-07 |
| results/jlens_cascade | 38 | 35.2 MB | 2026-10-07 |
| results/mistral_factorial | 4 | 569.0 KB | 2026-10-06 |
| results/paper_audit | 2 | 48.6 KB | 2026-10-05 |
| results/power_analysis_uu | 1 | 5.5 KB | 2026-10-05 |
| results/qwen_factorial | 5 | 573.5 KB | 2026-10-07 |
| results/rerun_analysis | 5 | 198.9 KB | 2026-10-05 |
| results/rerun_analysis_keyset_pooled | 3 | 41.0 KB | 2026-10-05 |
| results/second_model | 259 | 45.2 MB | 2026-10-05 |
| results/seed_sweep | 10 | 60.5 KB | 2026-10-05 |
| results/singlebos | 24 | 3.3 MB | 2026-10-07 |
| results/unireps_checks | 23 | 1.6 MB | 2026-10-07 |

## Provenance sidecars

- Files with 'provenance' in the name: 258
- results: 33
- results/second_model/qwen25_7b_instruct/results: 18
- results/circuit_familiarity_v3: 16
- results/second_model/llama31_base/results: 14
- results/circuit_familiarity_v2: 12
- results/circuit_conflict: 8
- results/circuit_familiarity: 8
- results/circuit_uu_hedge_confab_v3: 7

## Results folders that no .md or .tex file mentions

A folder is listed when neither its path nor (for names of 4+ characters) its bare name appears in any scanned text file.

- results/ablation_bf16_direction_familiarity (7 files)
- results/ablation_bf16_direction_familiarity_pooled (7 files)
- results/ablation_bf16_entropy_weights (8 files)
- results/ablation_bf16_entropy_weights_pooled (8 files)
- results/ablation_bf16_familiarity_new (5 files)
- results/ablation_bf16_familiarity_v2_dir (3 files)
- results/ablation_bf16_familiarity_v2_key (3 files)
- results/ablation_bf16_familiarity_v2_new (3 files)
- results/ablation_bf16_frequency_weights (8 files)
- results/bridge_familiarity_v2 (6 files)
- results/circuit_uu_hedge_confab_v3 (28 files)
- results/frequency_causal_v1_raw_readout (4 files)
- results/gemma_factorial (4 files)
- results/mistral_factorial (4 files)
- results/rerun_analysis_keyset_pooled (3 files)
- results/seed_sweep (10 files)

## Documents mentioned but absent on disk

A name is listed when no file with that path or base name exists anywhere in the scanned tree. It may be a generated, ignored, renamed or planned file.

### Documents (11)

- EXPLAINER.md (4 mentions, first at commit 2448363:1)
- l5_behavioral_validation.md (4 mentions, first at l5_behavioral_criterion.md:4)
- factorial_circuit_level_test.md (3 mentions, first at results/factorial_circuit_result.txt:5)
- uu_falsification_test.md (3 mentions, first at commit bdece64:2)
- regenerate_final_numbers.md (2 mentions, first at results/paper_audit/regenerate_final_numbers_result.txt:4)
- _summary.txt (1 mention, first at commit bdece64:12)
- dd.md (1 mention, first at paper/DD_REVIEW.md:1)
- familiarity_regeneration_plan.md (1 mention, first at commit e527a99:2)
- litreview/G_cross_model_convergence.md (1 mention, first at results/qwen_factorial/FINDINGS.md:47)
- paper/dd.md (1 mention, first at paper/DD_REVIEW.md:3)
- shshk/EXPERIMENT_PLAN.md (1 mention, first at paper/reviews_interpscience.md:23)

### Data or code files (11)

- _cells.csv (1 mention, first at commit 6e235d9:6)
- _hedgematched.csv (1 mention, first at commit e6cb649:8)
- _hedgematched.py (1 mention, first at commit 110e210:8)
- approval_tracker.jsonl (1 mention, first at archive/PIPELINES_README.md:8)
- causal_timing.csv (1 mention, first at commit d163fdc:13)
- cos_permutation.csv/.json (1 mention, first at commit 9ba67da:7)
- llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt (1 mention, first at results/jlens_bos_audit/bos_audit_familiarity.txt:1)
- mechanism_check_shared_old.json (1 mention, first at commit 86e370c:1)
- results/factorial_v3.csv (1 mention, first at commit d163fdc:18)
- shared/rescue_untruncated_candidates.py (1 mention, first at archive/notebooks/README.md:15)
- tokenizer.json (1 mention, first at results/unireps_checks/FINDINGS.md:3)

## File tree (depth 3)

```
analysis/  (2 files in total)
  stolfo_criteria.py
  token_frequency_neurons.py
archive/  (45 files in total)
  analysis/  (2 files in total)
    overlap_analysis.py
    overlap_result.txt
  notebooks/  (7 files in total)
    A_marimo_nb.py
    A_marimo_nb_cleaned.py
    B_marimo_nb.py
    B_marimo_nb_cleaned.py
    C_marimo_nb.py
    C_marimo_nb_cleaned.py
    README.md
  person_A_ambiguity/  (7 files in total)
    results/  (7 files in total)
  person_B_lack_of_knowledge/  (6 files in total)
    results/  (6 files in total)
  person_C_contradictory_context/  (7 files in total)
    results/  (7 files in total)
  pipelines/  (9 files in total)
    A_build_dataset.py
    A_old_preprocess_ambiguity_v2.py
    A_preprocess_ambiguity_v1.py
    B_build_dataset.py
    B_old_preprocess_lack_of_knowledge.py
    B_preprocess_lack_of_knowledge_v2.py
    C_build_dataset.py
    C_diagnose_templates.py
    C_old_preprocess_contradictory_context.py
  results/  (4 files in total)
    mechanism_check_shared.json
    old_mechanism_check_shared.json
    test_l29_category_difference.py
    test_merge.py
  PIPELINES_README.md
  README.md
  stolfo_criteria.csv
data/  (169 files in total)
  aleatoric/  (6 files in total)
    all_measured.jsonl
    candidates.jsonl
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  ambiguity/  (5 files in total)
    .gitkeep
    control_review_log.jsonl
    controls.jsonl
    prompts.jsonl
    review_log.jsonl
  conflict/  (9 files in total)
    pre_memory_guard/  (2 files in total)
    all_measured.jsonl
    candidates.jsonl
    consistently_wrong.jsonl
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  conflict_hedgegate/  (7 files in total)
    all_measured.jsonl
    candidates.jsonl
    consistently_wrong.jsonl
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  contradictory_context/  (5 files in total)
    .gitkeep
    controls.jsonl
    knows_fact_log.jsonl
    prompts.jsonl
    provenance.json
  factorial_v2_know_given_hi/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v2_know_given_lo/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v2_pop_given_known/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v2_pop_given_unknown/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_gemma_know_given_hi/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_gemma_know_given_lo/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_gemma_pop_given_known/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_gemma_pop_given_unknown/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_know_given_hi/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_know_given_lo/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_mistral_know_given_hi/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_mistral_know_given_lo/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_mistral_pop_given_known/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_mistral_pop_given_unknown/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_pop_given_known/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_pop_given_unknown/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_qwen_know_given_hi/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_qwen_know_given_lo/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_qwen_pop_given_known/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v3_qwen_pop_given_unknown/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  familiarity/  (5 files in total)
    consistently_wrong.jsonl
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  familiarity_v2/  (6 files in total)
    all_measured.jsonl
    consistently_wrong.jsonl
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  familiarity_v3/  (8 files in total)
    all_measured.jsonl
    consistently_wrong.jsonl
    controls.jsonl
    gate_report.json
    hedge_logodds.jsonl
    hedge_logodds.provenance.json
    prompts.jsonl
    prompts.provenance.json
  familiarity_v3_gemma/  (3 files in total)
    all_measured.jsonl
    hedge_logodds.jsonl
    hedge_logodds.provenance.json
  familiarity_v3_mistral/  (5 files in total)
    all_measured.jsonl
    gate_report.json
    hedge_logodds.jsonl
    hedge_logodds.provenance.json
    prompts.provenance.json
  familiarity_v3_qwen/  (8 files in total)
    all_measured.jsonl
    consistently_wrong.jsonl
    controls.jsonl
    gate_report.json
    hedge_logodds.jsonl
    hedge_logodds.provenance.json
    prompts.jsonl
    prompts.provenance.json
  lack_of_knowledge/  (3 files in total)
    .gitkeep
    controls.jsonl
    prompts.jsonl
  situated/  (6 files in total)
    all_measured.jsonl
    candidates.jsonl
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  uu_hedge_confab/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  uu_hedge_confab_v3/  (4 files in total)
    controls.jsonl
    gate_report.json
    prompts.jsonl
    prompts.provenance.json
  factorial_v2_summary.csv
  factorial_v3_gemma_summary.csv
  factorial_v3_mistral_summary.csv
  factorial_v3_qwen_summary.csv
  factorial_v3_summary.csv
figures/  (5 files in total)
  fig1.svg
  fig2.svg
  fig3.svg
  fig4.svg
  fig5.svg
labels_dr/  (8 files in total)
  gemma2b_base.jsonl
  gemma2b_inst.jsonl
  gemma9b_base.jsonl
  gemma9b_inst.jsonl
  llama_base.jsonl
  llama_inst.jsonl
  mistral_inst.jsonl
  qwen_inst.jsonl
notebooks/  (1 files in total)
  00_hello_world.ipynb
paper/  (26 files in total)
  figures/  (15 files in total)
    fig1.pdf
    fig2.pdf
    fig3.pdf
    fig4.pdf
    fig5.pdf
    fig6.pdf
    fig6_familiarity.pdf
    fig6_familiarity_cells.csv
    fig6_familiarity_v3.pdf
    fig6_familiarity_v3_cells.csv
    fig6_familiarity_v3_full.pdf
    fig6_familiarity_v3_full_cells.csv
    ... +3 more files
  figures_singlebos/  (4 files in total)
    fig4.pdf
    fig4.svg
    fig5.pdf
    fig5.svg
  DD_REVIEW.md
  DOC_MEMO.md
  LITREVIEW.md
  main.tex
  numbers.tex
  references.bib
  reviews_interpscience.md
person_A_ambiguity/  (5 files in total)
  notebooks/  (1 files in total)
    .gitkeep
  results/  (2 files in total)
    control_results_v3.csv
    results_v3.csv
  README.md
  preprocess_ambiguity.py
person_B_lack_of_knowledge/  (9 files in total)
  notebooks/  (1 files in total)
    .gitkeep
  results/  (2 files in total)
    control_results_v3.csv
    results_v3.csv
  README.md
  nec_templates.py
  rebuild_lack_of_knowledge_whitelisted.py
  screen_templates.py
  template_quota_report.json
  template_screen_results.json
person_C_contradictory_context/  (6 files in total)
  notebooks/  (1 files in total)
    .gitkeep
  results/  (2 files in total)
    control_results_v3.csv
    results_v3.csv
  LIMITATIONS.md
  README.md
  preprocess_contradictory_context.py
results/  (1054 files in total)
  ablation_bf16_conflict_key/  (4 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_conflict.csv.gz
    results_conflict.provenance.json
  ablation_bf16_conflict_new/  (4 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_conflict.csv.gz
    results_conflict.provenance.json
  ablation_bf16_direction_familiarity/  (7 files in total)
    ablation_means.json
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_direction_familiarity_pooled/  (7 files in total)
    ablation_means.json
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_entropy_weights/  (8 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_entropy_weights_pooled/  (8 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_familiarity_dir/  (4 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_familiarity.csv.gz
    results_familiarity.provenance.json
  ablation_bf16_familiarity_key/  (4 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_familiarity.csv.gz
    results_familiarity.provenance.json
  ablation_bf16_familiarity_new/  (5 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_familiarity.csv.gz
    results_familiarity.provenance.json
    specificity_random_null.csv
  ablation_bf16_familiarity_v2_dir/  (3 files in total)
    ablation_means.json
    results_familiarity_v2.csv
    results_familiarity_v2.provenance.json
  ablation_bf16_familiarity_v2_key/  (3 files in total)
    ablation_means.json
    results_familiarity_v2.csv
    results_familiarity_v2.provenance.json
  ablation_bf16_familiarity_v2_new/  (3 files in total)
    ablation_means.json
    results_familiarity_v2.csv
    results_familiarity_v2.provenance.json
  ablation_bf16_frequency_weights/  (8 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_keyset_pooled/  (8 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_new/  (8 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_old15/  (8 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_bf16_v3set/  (8 files in total)
    ablation_means.json
    entropy_adjusted.csv
    results_ambiguity.csv
    results_ambiguity.provenance.json
    results_contradictory_context.csv
    results_contradictory_context.provenance.json
    results_lack_of_knowledge.csv
    results_lack_of_knowledge.provenance.json
  ablation_conf_ctrl50/  (4 files in total)
    ablation_means.json
    results_conflict.csv.gz
    results_conflict.provenance.json
    specificity_random_null.csv
  ablation_fam_ctrl100/  (4 files in total)
    ablation_means.json
    results_familiarity.csv.gz
    results_familiarity.provenance.json
    specificity_random_null.csv
  bridge_familiarity/  (6 files in total)
    direction_bridge_directions.npz
    direction_bridge_neurons.csv
    direction_bridge_neurons.provenance.json
    direction_bridge_steer_familiarity.csv
    direction_bridge_steer_hedging.csv
    direction_bridge_summary.txt
  bridge_familiarity_v2/  (6 files in total)
    direction_bridge_directions.npz
    direction_bridge_neurons.csv
    direction_bridge_neurons.provenance.json
    direction_bridge_steer_familiarity.csv
    direction_bridge_steer_hedging.csv
    direction_bridge_summary.txt
  bridge_familiarity_v3/  (12 files in total)
    cos_bootstrap.csv
    cos_bootstrap.json
    cos_bootstrap.provenance.json
    cos_permutation.csv
    cos_permutation.json
    cos_permutation.provenance.json
    direction_bridge_directions.npz
    direction_bridge_neurons.csv
    direction_bridge_neurons.provenance.json
    direction_bridge_steer_familiarity.csv
    direction_bridge_steer_hedging.csv
    direction_bridge_summary.txt
  bridge_familiarity_v4/  (10 files in total)
    cos_bootstrap.csv
    cos_bootstrap.json
    cos_bootstrap.provenance.json
    cos_permutation.csv
    cos_permutation.json
    cos_permutation.provenance.json
    direction_bridge_neurons.provenance.json
    direction_bridge_steer_familiarity.csv
    direction_bridge_steer_hedging.csv
    direction_bridge_summary.txt
  bridge_familiarity_v5/  (12 files in total)
    cos_bootstrap.csv
    cos_bootstrap.json
    cos_bootstrap.provenance.json
    cos_permutation.csv
    cos_permutation.json
    cos_permutation.provenance.json
    direction_bridge_directions.npz
    direction_bridge_neurons.csv
    direction_bridge_neurons.provenance.json
    direction_bridge_steer_familiarity.csv
    direction_bridge_steer_hedging.csv
    direction_bridge_summary.txt
  circuit_aleatoric/  (4 files in total)
    jlens_direction_tokens.txt
    jlens_summary.txt
    jlens_trajectory.csv
    jlens_trajectory.provenance.json
  circuit_aleatoric_qwen/  (12 files in total)
    decider_expB_decomp.csv
    decider_expB_pair_decomp.csv
    decider_expB_sanity.csv
    decider_expB_token_breakdown.csv
    decider_expB_top_tokens.csv
    decider_expC_completions.csv
    decider_expC_summary.csv
    decider_provenance.json
    jlens_direction_tokens.txt
    jlens_summary.txt
    jlens_trajectory.csv
    jlens_trajectory.provenance.json
  circuit_conflict/  (33 files in total)
    direction_patch.csv
    direction_patch.provenance.json
    direction_patch_summary.txt
    direction_patch_u2c_hedgematched.csv
    direction_patch_u2c_hedgematched_diag.csv
    direction_patch_u2c_null.csv
    direction_patch_u2c_null.provenance.json
    faithfulness.csv
    faithfulness.provenance.json
    faithfulness_r10.csv
    faithfulness_summary.txt
    head_analysis.csv
    ... +21 more files
  circuit_conflict_qwen/  (14 files in total)
    direction_patch.csv
    direction_patch.provenance.json
    direction_patch_summary.txt
    jlens_demo.json
    jlens_direction_tokens.txt
    jlens_entity.csv.gz
    jlens_entity.provenance.json
    jlens_entity_summary.txt
    jlens_steer.csv
    jlens_steer.provenance.json
    jlens_steer_summary.txt
    jlens_summary.txt
    ... +2 more files
  circuit_factorial_v3_know_given_hi/  (23 files in total)
    behavioral.csv
    behavioral.provenance.json
    behavioral_suffix.csv
    behavioral_suffix.provenance.json
    behavioral_suffix_summary.txt
    behavioral_summary.txt
    faithfulness.csv
    faithfulness.provenance.json
    faithfulness_summary.txt
    head_analysis.csv
    head_attribution.csv
    head_verification.csv
    ... +11 more files
  circuit_factorial_v3_pop_given_unknown/  (23 files in total)
    behavioral.csv
    behavioral.provenance.json
    behavioral_suffix.csv
    behavioral_suffix.provenance.json
    behavioral_suffix_summary.txt
    behavioral_summary.txt
    faithfulness.csv
    faithfulness.provenance.json
    faithfulness_summary.txt
    head_analysis.csv
    head_attribution.csv
    head_verification.csv
    ... +11 more files
  circuit_familiarity/  (35 files in total)
    direction_patch.csv
    direction_patch.provenance.json
    direction_patch_summary.txt
    direction_patch_u2c_null.csv
    faithfulness.csv
    faithfulness.provenance.json
    faithfulness_r10.csv
    faithfulness_summary.txt
    head_analysis.csv
    head_attribution.csv
    head_verification.csv
    jlens_demo.json
    ... +23 more files
  circuit_familiarity_v2/  (49 files in total)
    causal_timing_from.csv
    causal_timing_from.provenance.json
    causal_timing_from_summary.csv
    causal_timing_from_summary.txt
    causal_timing_up_to.csv
    causal_timing_up_to.provenance.json
    causal_timing_up_to_suffix.csv
    causal_timing_up_to_suffix.provenance.json
    causal_timing_up_to_suffix_summary.csv
    causal_timing_up_to_suffix_summary.txt
    causal_timing_up_to_summary.csv
    causal_timing_up_to_summary.txt
    ... +37 more files
  circuit_familiarity_v3/  (62 files in total)
    behavioral.csv
    behavioral.provenance.json
    behavioral_result.txt
    behavioral_suffix.csv
    behavioral_suffix.provenance.json
    behavioral_suffix_summary.txt
    behavioral_summary.txt
    causal_timing_up_to_suffix.csv
    causal_timing_up_to_suffix.provenance.json
    causal_timing_up_to_suffix_summary.csv
    causal_timing_up_to_suffix_summary.txt
    direction_patch.csv
    ... +50 more files
  circuit_uu_hedge_confab_v3/  (28 files in total)
    behavioral.csv
    behavioral.provenance.json
    behavioral_suffix.csv
    behavioral_suffix.provenance.json
    behavioral_suffix_summary.txt
    behavioral_summary.txt
    faithfulness.csv
    faithfulness.provenance.json
    faithfulness_summary.txt
    faithfulness_working.csv
    faithfulness_working.provenance.json
    faithfulness_working_summary.txt
    ... +16 more files
  direction_nohedge/  (5 files in total)
    llama/  (4 files in total)
    FINDINGS.md
  direction_reuse/  (41 files in total)
    gemma2b/  (8 files in total)
    gemma9b/  (8 files in total)
    llama/  (8 files in total)
    mistral/  (8 files in total)
    qwen/  (8 files in total)
    FINDINGS.md
  direction_reuse_v2/  (11 files in total)
    gemma9b/  (5 files in total)
    llama/  (5 files in total)
    FINDINGS.md
  factorial_4models/  (7 files in total)
    FINDINGS.md
    cells.csv
    cross_model_items.csv
    cross_model_items_gemma.csv
    cross_model_items_mistral.csv
    effects.csv
    summary.txt
  frequency_causal_v1_raw_readout/  (4 files in total)
    frequency_causal.csv.gz
    frequency_causal.provenance.json
    frequency_causal_summary.csv
    frequency_causal_summary.txt
  gemma_factorial/  (4 files in total)
    cells.csv
    cross_model_items_gemma.csv
    effects.csv
    summary.txt
  hedge_judge/  (7 files in total)
    FINDINGS.md
    corrected_cells.csv
    judgments.jsonl
    judgments.provenance.json
    sample.jsonl
    stratum_counts.json
    summary.txt
  hedge_measure_v2/  (22 files in total)
    FINDINGS.md
    HUMAN_VALIDATION_README.md
    gens_gemma.jsonl
    gens_llama.jsonl
    gens_mistral.jsonl
    gens_qwen.jsonl
    human_validation_KEY_do_not_open_before_labelling.csv
    human_validation_sheet.csv
    item_rates.csv
    items_gemma.jsonl
    items_llama.jsonl
    items_mistral.jsonl
    ... +10 more files
  jlens_bos_audit/  (3 files in total)
    FINDINGS.md
    bos_audit_familiarity.csv
    bos_audit_familiarity.txt
  jlens_cascade/  (38 files in total)
    gemma-2-2b/  (4 files in total)
    gemma-2-2b-it/  (4 files in total)
    gemma-2-9b/  (4 files in total)
    gemma-2-9b-it/  (4 files in total)
    llama3.1-8b/  (4 files in total)
    llama3.1-8b-it/  (4 files in total)
    olmo-3-1025-7b/  (4 files in total)
    qwen2.5-7b-it/  (4 files in total)
    FINDINGS.md
    curves.csv
    fig_cascade.pdf
    fig_cascade.png
    outcomes.csv
    summary.txt
  mistral_factorial/  (4 files in total)
    cells.csv
    cross_model_items_mistral.csv
    effects.csv
    summary.txt
  paper_audit/  (2 files in total)
    main_tex_number_audit.csv
    regenerate_final_numbers_result.txt
  power_analysis_uu/  (1 files in total)
    power_analysis_uu_report.txt
  qwen_factorial/  (5 files in total)
    FINDINGS.md
    cells.csv
    cross_model_items.csv
    effects.csv
    summary.txt
  rerun_analysis/  (5 files in total)
    SUMMARY.md
    ablation_bf16_new_cells.csv
    ablation_bf16_old15_cells.csv
    ablation_bf16_v3set_cells.csv
    all_cells.csv
  rerun_analysis_keyset_pooled/  (3 files in total)
    SUMMARY.md
    ablation_bf16_keyset_pooled_cells.csv
    all_cells.csv
  second_model/  (259 files in total)
    llama31_base/  (91 files in total)
    qwen25_7b_instruct/  (168 files in total)
  seed_sweep/  (10 files in total)
    candidate_neurons_seed1.json
    candidate_neurons_seed1.provenance.json
    candidate_neurons_seed2.json
    candidate_neurons_seed2.provenance.json
    candidate_neurons_seed3.json
    candidate_neurons_seed3.provenance.json
    candidate_neurons_seed4.json
    candidate_neurons_seed4.provenance.json
    candidate_neurons_seed5.json
    candidate_neurons_seed5.provenance.json
  singlebos/  (24 files in total)
    circuit_conflict/  (4 files in total)
    circuit_conflict_qwen/  (4 files in total)
    circuit_familiarity/  (11 files in total)
    previews/  (4 files in total)
    FINDINGS.md
  unireps_checks/  (23 files in total)
    FINDINGS.md
    check1_decoupling_known.csv
    check1_familiarity_v3_gemma_decoupling_known.csv
    check1_familiarity_v3_gemma_items.csv
    check1_familiarity_v3_gemma_models.csv
    check1_familiarity_v3_gemma_pairs.csv
    check1_familiarity_v3_mistral_decoupling_known.csv
    check1_familiarity_v3_mistral_items.csv
    check1_familiarity_v3_mistral_models.csv
    check1_familiarity_v3_mistral_pairs.csv
    check1_familiarity_v3_qwen_decoupling_known.csv
    check1_familiarity_v3_qwen_items.csv
    ... +11 more files
  .gitkeep
  OVERVIEW_2026-10-07.md
  behavioral_bf16.csv
  behavioral_bf16_summary.csv
  candidate_neurons_bf16.json
  candidate_neurons_bf16.provenance.json
  candidate_vs_control.csv
  candidates_direction_familiarity.json
  candidates_entropy_weights.json
  candidates_frequency_weights.json
  candidates_keyset.json
  candidates_old15.json
  ... +127 more files
scripts/  (70 files in total)
  _common.py
  add_hedge_logodds.py
  aleatoric_leakage_corr.py
  analyze_rerun.py
  behavioral_test.py
  bridge_cosine_bootstrap.py
  bridge_cosine_permutation.py
  build_familiarity_twins.py
  build_frequency_knowledge_factorial.py
  build_uu_hedge_confab.py
  cells_to_candidates.py
  circuit_behavioral.py
  ... +58 more files
shared/  (12 files in total)
  __init__.py
  ablation.py
  baselines.py
  detection.py
  logit_lens.py
  model_utils.py
  old_detection.py
  prompt_format.py
  provenance.py
  run_ablation_pipeline.py
  run_phase34.py
  schema_utils.py
tests/  (10 files in total)
  test_behavioral_tiny.py
  test_cells_to_candidates_tiny.py
  test_circuit_tiny.py
  test_direction_bridge_tiny.py
  test_frequency_causal_tiny.py
  test_grader_tiny.py
  test_judge_audit_tiny.py
  test_multi_neuron_tiny.py
  test_reverse_nomination_tiny.py
  test_shared_tiny.py
.gitattributes
.gitignore
DATA_SOURCES.md
LICENSE
PHASE3_GUIDE.md
README.md
REFERENCES.md
RESULTS.md
RESULTS_SCHEMA.md
SETUP.md
UNCERTAINTY_DEFINITION.md
candidate_neurons.json
... +13 more files
```
