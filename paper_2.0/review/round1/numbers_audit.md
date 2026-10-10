# Numbers audit, round 1: main.tex against result files

Auditor scope: `paper/tex/main.tex`, `paper/numbers.tex` (macro values), the three figure provenance files, and the evidence files listed in the brief (R = `Confidence-Neurons-Across-Uncertainty-Types`). One further file was opened by following a path written in a numbers.tex comment: `R/paper/figures/fig6_familiarity_v3_full_cells.csv` (source of `\circuitHedgeCUVThreeFull` in the old `R/paper/numbers.tex`). No FINDINGS/notes files were opened. Values were recomputed from `item_rates.csv`, `cells.csv`, `check1_pairs.csv`, the three `steer.csv` files and `curves.csv` with a script; all summary-file values reproduced.

File dates: result files Oct 5 to Oct 7; `paper/numbers.tex` Oct 10 00:47; `main.tex` Oct 10 01:03; figures Oct 10 00:55 to 00:57. No macro was found stale relative to its result file. `R/data/familiarity_v3/gate_report.json` (Oct 5) is older than, and disagrees with, the Llama factorial pool (see M-9).

Status key: **exact** (identical, or a comparison/scope that the file supports as stated); **rounding-ok** (correct 2-dp rounding); **mismatch** (wrong value, wrong count, wrong configuration, or scope word not supported); **no-evidence-found** (no supporting value in the permitted files).

## Item table

| id | location | claim text (exact wording / macro → value) | evidence file and field | value in file | status |
|---|---|---|---|---|---|
| 1 | Abstract | "follow popularity (effect \hmLlamaPopEff)" → +0.39 | hedge_measure_v2/summary.txt, llama primary `pop` | +0.388 | rounding-ok |
| 2 | Abstract | "far more than knowledge (\hmLlamaKnowEff)" → +0.09 | same, primary `know` | +0.089 | rounding-ok |
| 3 | Abstract | "far more than" (comparison) | same | 0.388 vs 0.089 (4.4x), diff CI [+0.204,+0.398] | exact |
| 4 | Abstract | "the entropy pattern holds in four model families" | factorial_4models/summary.txt, condition 2 per model | "yes" for llama, qwen, gemma, mistral | exact |
| 5 | Abstract | "Steering ... makes Llama and Gemma-2 say IDK about facts they answer correctly" | direction_reuse_v2/{llama,gemma9b}/summary.txt `none`, `own_fam` | 0.020 -> 0.287; 0.000 -> 0.220 on popular HighlyKnown items | exact |
| 6 | Abstract | "mostly through components shared with abstention itself" | direction_nohedge/llama/summary.txt | d_nohedge_U - random +0.041 vs d_fam - random +0.261 (16%) | exact |
| 7 | Abstract | "familiarity signal at mid-depth in base models of three families" | jlens_cascade/outcomes.csv midF, base models | 4 base models (Llama, Gemma x2, OLMo), all midF CI > 0 | exact |
| 8 | Intro (i) | "(\hmLlamaPopEff) more than knowledge (\hmLlamaKnowEff)" → +0.39, +0.09 | hedge_measure_v2/summary.txt | +0.388, +0.089 | rounding-ok |
| 9 | Intro (i) | "answer uncertainty follows knowledge in all four families tested" | factorial_4models/effects.csv entropy | know > pop, CIs disjoint, in all 4 | exact |
| 10 | Intro (ii) | "makes Llama and Gemma-2-9B say IDK about popular facts they know" | direction_reuse_v2 summaries | see 5 | exact |
| 11 | Intro (iii) | "decodable at mid-depth in base models of three families, so it predates instruction tuning" | jlens_cascade/summary.txt "PREDATES INSTRUCTION TUNING ... yes"; jlens_cascade_criterion.md licence | yes | exact |
| 12 | Setup | "PopQA questions from 16 relations" | gate_report.json `n_relations`; unireps_checks/summary.txt "relations 16" | 16 | exact |
| 13 | Setup | "bottom and top deciles of Wikipedia page views" | scripts/build_familiarity_twins.py `--lo-quantile 0.10`, `--hi-quantile 0.90` on `s_pop` | 0.10 / 0.90 | exact (note: arms are a shuffled subset capped by `--n-per-relation`, not the whole decile) |
| 14 | Setup | "a greedy answer and ten samples ($T=0.7$)" | build_familiarity_twins.py `--n-samples 10`, `temperature=0.7`; gate_report `n_samples` 10 | 10, 0.7 | exact |
| 15 | Setup | "known if the greedy answer and at least 90% of samples are correct and unknown if none is" | build_familiarity_twins.py `slick_class` | greedy_ok and frac >= 0.9; frac == 0 and not greedy_ok | exact |
| 16 | Setup | "\faLlamaN{} questions for Llama" → 2,303 | factorial_4models/cells.csv llama sum of n | 1179+87+361+676 = 2303 (also unireps "items: 2303") | exact; but gate_report.json (familiarity_v3) gives HighlyKnown 758 + Unknown 1513 = 2271 (files disagree, see M-9) |
| 17 | Setup | \faQwenN → 2,392 | cells.csv qwen | 1209+54+741+388 = 2392 | exact |
| 18 | Setup | \faGemmaN → 2,282 | cells.csv gemma | 1129+122+410+621 = 2282 | exact |
| 19 | Setup | \faMistralN → 2,378 | cells.csv mistral | 1178+71+417+712 = 2378 | exact |
| 20 | Setup | "forced-answer prompt: the chat template plus ... 'The answer is'" | build_familiarity_twins.py docstring step 3 | chat template + prefill "The answer is" | exact |
| 21 | Setup | "Gemma: popularity \loGemmaPop" → +1.79 | effects.csv gemma hedge_logodds pop | +1.789 | rounding-ok |
| 22 | Setup | "knowledge \loGemmaKnow" → -0.32 | effects.csv gemma hedge_logodds know | -0.318 | rounding-ok |
| 23 | Setup | "in Llama both effects are large" (offered as evidence the popularity signal survives the prompt) | effects.csv llama hedge_logodds | pop +2.933, know +3.055, pop-know -0.122 [-0.722,+0.449]; Qwen pop-know -2.494 [-3.413,-1.644]; Mistral +0.403 [-0.545,+1.344] | mismatch (selected value: only the one model where pop >> know is quantified; in Llama and Qwen the forced-answer log-odds do not favour popularity) |
| 24 | Setup | "the steering and lens analyses each use one prompt" | direction_reuse_criterion.md "E (entropy): first-token entropy at the prefill prompt"; direction_reuse_v2_criterion.md "Entropy is measured at the prefill prompt" | steering abstention on free prompt, steering entropy on forced-answer prompt | mismatch (configuration) |
| 25 | Setup | "For up to 100 questions per cell" | item_rates.csv cell counts; hedge_measure_v2_criterion.md | llama 100/100/100/87; criterion "100 items per cell (all if fewer)" | exact |
| 26 | Setup | "(\hmLlamaItems{} in total)" → 387 | hedge_measure_v2/summary.txt; item_rates.csv | 387 | exact |
| 27 | Setup | "three free replies ($T=0.7$, up to 96 tokens)" | hedge_measure_v2_criterion.md Decoding; summary 1161 samples = 3 x 387 | 3, 0.7, 96 | exact |
| 28 | Setup | "Two LLM judges (Qwen2.5-32B-Instruct and Gemma-2-27B-it)" | hedge_measure_v2_criterion.md Labelling | same | exact |
| 29 | Setup | "abstains when both say DONT_KNOW" | criterion "Primary hedge measure" | same | exact |
| 30 | Setup | "$\kappa=\hmLlamaKappa$" → 0.62 | summary.txt llama kappa | 0.62 | exact |
| 31 | Setup | "just above the bar of \kappaBar{} fixed before the run" → 0.60 | hedge_measure_v2_criterion.md "κ ≥ 0.60" | 0.60 | exact (criterion file mtime Oct 7 00:48 equals items_*.jsonl mtime 00:48; "before the run" not verifiable from file dates; no hash in evidence) |
| 32 | Setup | "($\kappa=\hmQwenKappa$, \hmGemmaKappa, \hmMistralKappa)" → 0.30, 0.47, 0.39 | summary.txt kappa per model | 0.30, 0.47, 0.39 | exact |
| 33 | Setup | "these models rarely abstain" | hedge_measure_v2 obscure/Unknown: primary 0.050/0.047/0.097; secondary 0.317/0.220/0.253. hedge_judge/summary.txt judge-corrected obscure/Unknown 0.764/0.420/0.736 | 5-10% (primary) vs 22-32% (secondary) vs 42-76% (single judge, 24-token) | mismatch (scope word holds only for the primary measure; two evidence files disagree) |
| 34 | Setup | "abstention is reported for Llama only" | criterion "If κ < 0.60, ... reported as unmeasured" | consistent | exact |
| 35 | Setup | "with 95% intervals from 10,000 item bootstraps within cells" | hedge_measure_v2_criterion.md (10,000 within-cell); factorial effect CIs: method in qwen_factorial_criterion.md (not in evidence); lens: criterion gives 2,000 for D, count unstated for midF; steering: "prompt-bootstrap", count unstated | 10,000 only documented for the hedge measure | no-evidence-found (for the entropy, steering and lens intervals) |
| 36 | Setup | "criterion note hashed before its run" | none of the permitted files contains hashes | - | no-evidence-found |
| 37 | Fig 1 caption | "($n=\hmLlamaItems$ questions, 3 replies each)" → 387 | fig_dissociation.provenance.json panel a n: 87+100+100+100 | 387 | exact |
| 38 | Fig 1 caption | "(b) ... ($n=\faLlamaN$)" → 2,303 | provenance panel b n: 87+1179+676+361 | 2303 | exact |
| 39 | Fig 1 caption | "(c) Popularity and knowledge effects on that entropy in four families" | provenance c/* vs effects.csv entropy | identical to full precision for all 4 | exact |
| 40 | Fig 1 panel a (plotted) | cell means | provenance a/* vs hedge_measure_v2 cells | 0.4767, 0.4330, 0.1333, 0.0 | exact |
| 41 | Fig 1 panel b (plotted) | cell means | provenance b/* vs cells.csv llama entropy | 3.510, 0.951, 2.577, 0.858 | exact |
| 42 | Fig 1 caption | "Means with 95% item-bootstrap intervals" | provenance CIs present; resample count not recorded | - | exact (method partly recoverable) |
| 43 | §3.1 | "in \hmLlamaObsUnk{} of replies when it lacks the fact" → 0.48 | summary primary unc/Unkn | 0.477 | rounding-ok |
| 44 | §3.1 | "\hmLlamaObsKnown{} when it knows it" → 0.43 | primary unc/High | 0.433 | rounding-ok |
| 45 | §3.1 | "for popular entities, in \hmLlamaPopUnk" → 0.13 | primary con/Unkn | 0.133 | rounding-ok |
| 46 | §3.1 | "and \hmLlamaPopKnown" → 0.00 | primary con/High | 0.000 | exact |
| 47 | §3.1 | "popularity effect (+0.39 [+0.33, +0.45])" | primary pop | +0.388 [+0.328,+0.449] | rounding-ok |
| 48 | §3.1 | "knowledge effect (+0.09 [+0.03, +0.15])" | primary know | +0.089 [+0.031,+0.147] | rounding-ok |
| 49 | §3.1 | "by +0.30 [+0.20, +0.40]" | primary pop-know | +0.300 [+0.204,+0.398] | rounding-ok |
| 50 | §3.1 | "Knowledge still matters, at about a quarter of the popularity effect" | recomputed from item_rates.csv | 0.0885 / 0.3881 = 0.228 | rounding-ok |
| 51 | §3.1 | "keeps its sign under three other measurements" | Table A lists primary + 4 other measurements, all positive | 4 others | mismatch (count) |
| 52 | §3.1 | "a regex detector with recall \hmLlamaRegexRecall{} on these replies" → 0.99 | summary llama "regex recall on 96-token replies = 0.99" | 0.99 | exact (on the 24-token replies the same regex has recall 0.80: hedge_judge/summary.txt) |
| 53 | §3.1 | "entropy rises by \entLlamaKnow{} nats when Llama lacks the fact" → +2.14 | effects.csv llama entropy know | 2.1390 | rounding-ok |
| 54 | §3.1 | "and by \entLlamaPop{} when the entity is obscure" → +0.51 | effects.csv llama entropy pop | 0.5128 | rounding-ok |
| 55 | §3.1 | "In every family the knowledge effect is at least four times the popularity effect" | effects.csv entropy know/pop | Llama 4.17, Qwen pop -0.019 (ratio undefined, |65|), Gemma 11.6, Mistral 4.79 | exact (point estimates; Llama is 4.17, and 3.37 at the CI bounds 2.045/0.606) |
| 56 | §3.1 | "a regression over the full Llama pool" | unireps_checks/summary.txt "items: 2303 (HighlyKnown 763, Unknown 1540)" | 2,303 labelled items, not the 2,673-item pool | mismatch (scope wording, minor) |
| 57 | §3.1 | "with relation fixed effects" | summary.txt "relation fixed effects; SE clustered by relation" | yes | exact |
| 58 | §3.1 | "at \scPoolPop{} logits per standard deviation" → -1.87 | summary.txt hedge GLM M1 logpop_z | -1.871 [-2.355,-1.388] | rounding-ok |
| 59 | §3.1 | "at \scPoolPopSurface{} after controlling for the name's word frequency, token count and token rarity" → -1.79 | M2 logpop_z; M2 covariates zipf_mean, zipf_min, n_tok, tok_logid | -1.791 | rounding-ok |
| 60 | §3.1 | "rules out name spelling as the main driver" | M1 -1.871 vs M2 -1.791 | 4% shrinkage | exact |
| 61 | Fig 2 caption | "on \stLlamaN{} held-out popular, known questions" → 150 | both v2 summaries n = 150; steer.csv 150 unique keys each | 150 (Gemma also 150; \stGemmaN unused) | exact |
| 62 | Fig 2 caption | "regex detector, which undercounts Gemma's abstentions" | hedge_measure_v2 gemma regex recall 0.48; hedge_judge gemma recall 0.14 | - | exact |
| 63 | Fig 2 caption | "dots: individual random directions" | fig_steering.provenance.json records only `random` mean 0.02533 and its CI; individual values exist in summary.txt (0.027,0.020,0.040,0.013,0.027) | not recorded as plotted | no-evidence-found |
| 64 | Fig 2 caption | "hatched: the direction fitted only on never-abstained questions" | provenance a/llama/d_nohedge_U 0.0667 | d_nohedge_U | exact |
| 65 | Fig 2 caption | "Error bars: 95% bootstrap over questions" | provenance CIs; criterion "95% prompt-bootstrap"; resample count not stated | - | exact (count unrecoverable) |
| 66 | Fig 2 plotted | all bars | provenance vs steer.csv recomputation | 0.02, 0.0253, 0.0867, 0.2867, 0.0667; entropy 0.824, 0.895, 1.225, 1.256; Gemma 0, 0, 0, 0.22; 0.254, 0.534, 1.500, 0.819 | exact |
| 67 | §3.2 | "On a random half of each model's questions" | direction_reuse_criterion.md "fixed random half (seed 0)" | yes | exact |
| 68 | §3.2 | "at the last position of the free prompt" | direction_reuse_v2_criterion.md change 1 | yes | exact |
| 69 | §3.2 | "difference of cell means, averaged over knowledge" | direction_reuse_criterion.md d_fam definition | ½[(oU-pU)+(oHK-pHK)] | exact |
| 70 | §3.2 | "layer \stLlamaLayer{} of \stLlamaLayers" → 17 of 32 | llama summary "l* = 17 of 32"; criterion round(0.55 x 31) = 17 | 17, 32 | exact |
| 71 | §3.2 | "\stGemmaLayer{} of \stGemmaLayers" → 23 of 42 | gemma summary "l* = 23 of 42"; round(0.55 x 41) = 23 | 23, 42 | exact |
| 72 | §3.2 | "at the last prompt position and every generated one" | criterion Intervention | yes | exact |
| 73 | §3.2 | "including five random ones" | summary random0-4 | 5 | exact |
| 74 | §3.2 | "a dose chosen on the obscure-minus-popular direction alone (\stLlamaDose$\times$ and \stGemmaDose$\times$)" → 1x, 4x | summaries "k* = 1", "k* = 4"; v2 criterion "smallest k at which own - random ≥ 0.10"; gemma log dose1 +0.000, dose2 +0.007, dose4 +0.220 | 1, 4 | exact |
| 75 | §3.2 | "from \stLlamaNone{} to \stLlamaFam" → 0.02 to 0.29 | llama summary none, own_fam | 0.020, 0.287 | rounding-ok |
| 76 | §3.2 | "against \stLlamaKnow{} for unknown-minus-known" → 0.09 | inst_know | 0.087 | rounding-ok |
| 77 | §3.2 | "and \stLlamaRand{} for random directions" → 0.03 | random mean | 0.025 (0.02533 from steer.csv) | rounding-ok |
| 78 | §3.2 | "In Gemma it rose from \stGemmaNone{} to \stGemmaFam" → 0.00 to 0.22 | gemma none, own_fam | 0.000, 0.220 | exact |
| 79 | §3.2 | "while no control moved it" | gemma inst_know, base_fam, random0-4 | all 0.000 | exact |
| 80 | §3.2 | "unknown-minus-known direction instead raised answer entropy more (difference \stGemmaDE{} \stGemmaDECI)" → +0.68 [+0.59, +0.77] | gemma DD entropy(know)-entropy(fam) | +0.681 [+0.593,+0.770] | rounding-ok (entropy was measured on the forced-answer prompt; see 24) |
| 81 | §3.2 | "a double dissociation" | gemma DD "-> yes"; criterion rule | yes | exact |
| 82 | §3.2 | "in Llama both raised entropy similarly (\stLlamaDE{} \stLlamaDECI)" → -0.03 [-0.11, +0.05] | llama DD entropy | -0.031 [-0.110,+0.047] | rounding-ok |
| 83 | §3.2 | "raised abstention by just \nhOverRand{} \nhOverRandCI{} over random directions" → +0.04 [+0.01, +0.07] | direction_nohedge summary "d_nohedge_U - random +0.041 [+0.013, +0.073]" | +0.041 [+0.013,+0.073] | rounding-ok |
| 84 | §3.2 | "A version fitted only on questions Llama never abstained on" (which version) | direction_nohedge log/summary: two such directions, d_nohedge_U 0.067 and d_nohedge_both 0.087 (both - random = +0.061) | paper quotes the U version without saying it differs in construction from d_fam (cos 0.605 vs 0.853 for "both") | mismatch (configuration not stated; minor) |
| 85 | §3.2 | "most of the effect runs through components shared with abstaining" | 0.041 / 0.261 = 16%; with d_nohedge_both 0.061 / 0.261 = 24% | either way < 50% | exact |
| 86 | Fig 3 caption | "for four base models" | fig_lens_base.provenance.json | llama3.1-8b, gemma-2-9b, gemma-2-2b, olmo-3-1025-7b | exact |
| 87 | Fig 3 caption | "Bands: 95% item-bootstrap intervals" | curves.csv F_lo/F_hi/K_lo/K_hi; count not stated | - | exact (count unrecoverable) |
| 88 | Fig 3 caption | "shaded: the mid-depth window fixed before the runs" | jlens_cascade_criterion.md [1/3, 2/3]; criterion mtime Oct 6 15:35, curves Oct 6 23:51 | [1/3, 2/3] | exact |
| 89 | §3.3 | "We decoded every layer of four base models" | curves.csv layers 0..n_layers-2 (e.g. 0-30 for 32-layer models) | final layer absent | mismatch (minor wording) |
| 90 | §3.3 | "end of 'Question: ... Answer: The answer is'" | jlens_cascade_criterion.md base prompt | same | exact |
| 91 | §3.3 | "In all four models the obscure-minus-popular effect is positive in the pre-set mid-depth window" | outcomes.csv midF base | 0.671, 0.387, 0.401, 0.293 | exact |
| 92 | §3.3 | "from \jlOlmoBaseMidF{} \jlOlmoBaseMidFCI{} in OLMo-3-7B" → +0.29 [+0.19, +0.38] | outcomes.csv olmo midF | 0.2931 [0.1911, 0.3844] | rounding-ok |
| 93 | §3.3 | "to \jlLlamaBaseMidF{} \jlLlamaBaseMidFCI{} in Llama-3.1-8B" → +0.67 [+0.59, +0.75] | outcomes.csv llama3.1-8b | 0.6709 [0.5871, 0.7488] | rounding-ok |
| 94 | §3.3 | "Gemma-2-9B ... at \jlGemmaNineBaseMidF" → +0.39 | outcomes.csv gemma-2-9b | 0.3865 | rounding-ok |
| 95 | §3.3 | "and 2B at \jlGemmaTwoBaseMidF" → +0.40 | outcomes.csv gemma-2-2b | 0.4006 | rounding-ok |
| 96 | §3.3 | "all intervals exclude zero" | midF_lo base: 0.587, 0.345, 0.328, 0.191 | all > 0 | exact |
| 97 | §3.3 | "Across eight base and instruct models" | outcomes.csv rows; summary "8 of 8 planned" | 8 | exact |
| 98 | §3.3 | "\jlSupports{} supporting, \jlContradicts{} contradicting and \jlUndecided{} undecided" → 1, 1, 6 | jlens_cascade/summary.txt | SUPPORTS 1, CONTRADICTS 1, UNDECIDED 6 | exact |
| 99 | §3.3 | "the two signals emerge together" | outcomes.csv D: olmo -5 [-5,-5] (CONTRADICTS), gemma-2-9b +1 [+1,+2] (SUPPORTS), gemma-2-2b-it -2 [-2,0]; summary "CROSS-MODEL CLAIM: mixed" | D = 0 in only 5 of 8 | mismatch (scope) |
| 100 | §3.3 | "at relative depth \jlOnsetLo--\jlOnsetHi" → 0.45-0.74 | outcomes.csv min/max of rel_F, rel_K over 8 models | 0.4516 (olmo K), 0.7407 (qwen2.5-7b-it) | rounding-ok |
| 101 | §3.3 | "(\cref{fig:lens}b)" as the support for 0.45-0.74 | Fig 3 plots base models only; base onsets span 0.4516-0.64 | 0.74 comes from Qwen2.5-7B-Instruct, not plotted | mismatch (caption vs content) |
| 102 | Limitations | "Abstention is measured reliably in one model" | κ: only Llama ≥ 0.60 | yes | exact |
| 103 | Limitations | "the other three families rarely say IDK" | see 33 | 5-10% / 22-32% / 42-76% by measure | mismatch (scope; see 33) |
| 104 | Limitations | "their agreement on Llama only just clears its bar" | 0.62 vs 0.60 | yes | exact |
| 105 | Limitations | "although four measurements agree in sign" | Table A has five rows, all positive | 5 | mismatch (count) |
| 106 | Limitations | "on single-entity PopQA questions and 7--9B models" | jlens_cascade outcomes include gemma-2-2b and gemma-2-2b-it | 2B models in the lens study | mismatch (scope, minor) |
| 107 | Table A caption | "under four measurements" | table rows | 5 rows | mismatch (count) |
| 108 | Table A row 1 | "Regex detector: 20 per question, 24 tokens, \faLlamaN{} questions" → 2,303 | gate_report `n_free` 20; build_familiarity_twins.py `max_new_tokens=24`; unireps "free generations per item: 20" | 20, 24, 2303 | exact |
| 109 | Table A row 1 | "\faLlamaRegexDiff" → +0.44 | effects.csv llama hedge_rate diff | +0.4423 | rounding-ok |
| 110 | Table A row 2 | "One LLM judge, corrected: 24 tokens" | hedge_judge/summary.txt (Qwen2.5-32B, 1515 generations); hedge_measure_v2_criterion.md "24-token free generations" | 24 tokens | exact |
| 111 | Table A row 2 | "\hjLlamaDiff" → +0.13 | hedge_judge llama corrected pop - know | +0.132 [+0.027,+0.239] | rounding-ok |
| 112 | Table A row 3 | "3 per question, 96 tokens, \hmLlamaItems{} questions" → 387 | criterion; summary | 3, 96, 387 | exact |
| 113 | Table A row 3 | "\hmLlamaDiff{} \hmLlamaDiffCI" → +0.30 [+0.20, +0.40] | summary | +0.300 [+0.204,+0.398] | rounding-ok |
| 114 | Table A row 4 | "\hmLlamaSecDiff" → +0.31 | summary secondary pop-know | +0.311 | rounding-ok |
| 115 | Table A row 5 | "\hmLlamaRegexDiff" → +0.43 | summary regex pop-know | +0.429 | rounding-ok |
| 116 | Table A caption | "Positive values mean abstention follows popularity more than knowledge" | all five values positive | yes | exact |
| 117 | App A | "Among \scPopKnownPairs{} matched pairs of known facts" → 55 | check1_pairs.csv contrast pop_given_known | 55 | exact |
| 118 | App A | "the abstention gap of \scPopKnownGap" → +0.76 | check1_pairs.csv mean d_hedge | 0.7582 | rounding-ok |
| 119 | App A | "falls to \scPopKnownGapSurface{} \scPopKnownGapSurfaceCI" → +0.43 [+0.18, +0.69] | unireps summary "hedge gap at EQUAL surface form" | +0.434 [+0.182,+0.686] | rounding-ok (interval method of the regression intercept not stated in the file) |
| 120 | App A | "Some obscure names also contain their answer" | summary "obscure twin name-inferable 35%" | 35% (19 of 55) | exact |
| 121 | App A | "a gap of \scNonInferGap{} over \scNonInferN{} pairs" → +0.79 over 36 | summary "n=36, hedge gap +0.788"; recomputed 0.7875 | +0.788, 36 | rounding-ok |
| 122 | App A | detector used for 117-121 not named | unireps summary: free-generation regex (20 x 24 tokens) | regex, the superseded measure | exact (omission noted) |
| 123 | Table B | κ row 0.62, 0.30, 0.47, 0.39; bar 0.60 | summary.txt; criterion | same | exact |
| 124 | Table B | Both DK: 0.48, 0.05, 0.05, 0.10 | summary primary unc/Unkn | 0.477, 0.050, 0.047, 0.097 | rounding-ok |
| 125 | Table B | DK or CLARIFY: 0.32, 0.22, 0.25 | summary secondary unc/Unkn | 0.317, 0.220, 0.253 | rounding-ok |
| 126 | Table B | Regex: 0.18, 0.07, 0.21 | summary regex unc/Unkn | 0.180, 0.073, 0.210 | rounding-ok |
| 127 | Table B | regex rows reported without recall | hedge_measure_v2_criterion.md: "the regex verdicts are reported with their known recall"; recalls 0.58/0.48/0.73 (96-token), 0.10/0.14/0.14 (24-token) | not reported (\hjQwenRecall etc. defined but unused) | mismatch (criterion compliance, minor) |
| 128 | App C | "reaches half its final size at layer \ltHedgeHalf" → 16 | singlebos/circuit_familiarity/jlens_summary.txt | L16 | exact |
| 129 | App C | "the entropy gap does so at layer \ltEntHalf" → 28 | same | L28 | exact |
| 130 | App C | "\ltLead{} layers later" → 12 | 28 - 16 | 12 | exact |
| 131 | App C | "Neurons selected for their correlation with output entropy in Llama and Qwen behaved as word-frequency and temperature neurons" | no permitted file | - | no-evidence-found |
| 132 | App C | "20 attention heads and 100 MLP neurons" | circuit_familiarity_v3/behavioral_summary.txt "20 heads + 100 neurons" | 20, 100 | exact |
| 133 | App C | "reproduces \ciRecovery{} \ciRecoveryCI" → 0.83 [0.78, 0.87] | R/paper/numbers.tex \circuitHedgeCUVThreeFull(CI); fig6_familiarity_v3_full_cells.csv logodds_rec control_to_uncertain full set | 0.8272 [0.7842, 0.8714], n=141, bootstrap_pairs | rounding-ok (direction not stated; the other direction, uncertain_to_control, is 1.034 [0.995, 1.074]) |
| 134 | App C | "removing its state removes abstention in \ciRemoveFlip" → 0.98 | behavioral_summary control_to_uncertain circuit flip | 0.981 (103 discordant of 141 pairs) | rounding-ok |
| 135 | App C | "injecting it produces abstention in only \ciInjectFlip" → 0.40 | behavioral_summary uncertain_to_control circuit flip | 0.398; amended detector 0.481 (unireps check 3) | rounding-ok (two detector values exist) |
| 136 | App C | "fell short of the behavioural bar we had fixed in advance" | no bar in behavioral_summary.txt or other permitted files | - | no-evidence-found |
| 137 | App D | "the note's SHA-256 hash was recorded before the run" | no hash in permitted files | - | no-evidence-found |
| 138 | App D | criteria "cover the four-family design, the two-judge measurement, the steering experiment and its never-abstained control, and the lens study" | found: hedge_measure_v2, direction_reuse v1/v2, jlens_cascade criteria. Four-family (qwen_/gemma_mistral_factorial_criterion.md) and never-abstained criteria not in permitted files | 3 of 5 seen | no-evidence-found (for two of five) |
| 139 | App D | "instruct-versus-base comparison of lens magnitudes ... reported as post hoc" | no such comparison appears in main.tex; jl*Inst* macros unused; lens criterion: "Report each base and instruct pair side by side either way" | not reported | mismatch (orphan reference; criterion compliance) |
| 140 | Related work | "Gemma-2 and Claude 3.5 Haiku"; "across ten models" (citations) | not checkable against result files | - | no-evidence-found (citation; out of evidence scope) |

## Further checks

**Configuration mismatches between compared conditions**
- Steering compares abstention (free prompt, greedy, 24 tokens) with entropy (forced-answer prompt, vector added at its last position) under the same direction. The Gemma "double dissociation" and the Llama "similar entropy" result both cross prompts. Item 24.
- Fig 1(a) uses 387 items with 96-token replies and two judges; Fig 1(b) uses 2,303 items. The caption gives both n. OK.
- Candidate pool size is inconsistent across files: 2,673 (factorial_4models/summary.txt Llama; direction_reuse_criterion.md; steering logs "2673 items") vs 2,637 (factorial Qwen/Gemma/Mistral; both gate_report.json files; jlens_cascade_criterion.md, which says "the same 2,637 PopQA candidates as every factorial"). So the Llama factorial pool has 36 more candidates than the other three families.
- The lens study's instruct label sets differ from the factorial sets for Qwen (lens 1237/54/746/390 = 2,427 vs factorial 1209/54/741/388 = 2,392) and Gemma-it (lens 2,330 vs factorial 2,282). The lens criterion says Qwen "reuses its existing familiarity_v3* labels". No quoted number depends on this.

**Interval methods**
- Recoverable: hedge measure (10,000 within-cell item bootstrap, criterion); lens D (2,000 within-cell bootstraps, criterion); circuit recovery (`bootstrap_pairs`, n=141; count not stated).
- Not recoverable from permitted files: factorial entropy effect CIs, steering CIs (resample count), no-hedge CI, lens midF CIs (count), surface-form intercept CI (regression method). The Setup sentence attributes 10,000 bootstraps to all effects.

**Orphans** (63 of 146 macros defined in numbers.tex are not used in main.tex): hmLlamaSamples, faLlamaRegexPop, faLlamaRegexKnow, hjQwenRecall, hjGemmaRecall, hjMistralRecall, entLlamaKnowCI, entLlamaPopCI, ent{Qwen,Gemma,Mistral}{Pop,Know}{,CI}, hr{Llama,Qwen,Gemma,Mistral}ObsUnk, loLlamaPop, loLlamaKnow, stLlamaBase, stLlamaDH(CI), stLlamaEnt{None,Know,Fam}, stGemmaN, stGemmaKnow, stGemmaBase, stGemmaRand, stGemmaDH(CI), stGemmaEnt{None,Know,Fam}, nhNoHedge, nhCos, jl*BaseOutcome, jlGemmaNineBaseMidFCI, jlGemmaTwoBaseMidFCI, all jl*Inst* macros, poolCandidates, poolLlama. Spot-checked orphan values all match their files (e.g. hrLlamaObsUnk 0.849, stLlamaDH +0.200 [+0.140,+0.267], nhNoHedge 0.067, hjQwenRecall 0.10). One ambiguity: nhCos = 0.60 while log.txt prints +0.605. No macro used in main.tex is undefined.

**Stale values**: none found. All macros trace to files dated Oct 5 to 7 and reproduce them. `R/data/familiarity_v3/gate_report.json` (Oct 5) predates and disagrees with the Llama factorial pool (2,637 candidates / 2,271 labelled vs 2,673 / 2,303).

## Summary

Counts over 140 items: **exact 76, rounding-ok 41, mismatch 15, no-evidence-found 8**. The mismatches are items 23, 24, 33, 51, 56, 84, 89, 99, 101, 103, 105, 106, 107, 127 and 139.

### Critical mismatches (would change a stated conclusion or the argument that supports it)

- **C-1 (item 24): "the steering and lens analyses each use one prompt."** Both direction-reuse criteria state that steering entropy is measured at the forced-answer (prefill) prompt, while steering abstention is measured on the free prompt. So the steering analysis uses two prompts, and the Gemma double dissociation crosses the same prompt change the sentence is meant to rule out. *Correction:* say that steering measures abstention on the free prompt and entropy on the forced-answer prompt (direction_reuse_v2_criterion.md, "Entropy is measured at the prefill prompt"), and drop steering from the one-prompt defence.
- **C-2 (item 99): "the two signals emerge together."** outcomes.csv has OLMo-3 knowledge onset 5 layers before familiarity, D = -5 with CI [-5, -5] (CONTRADICTS). Gemma-2-9B has familiarity first, D = +1 [+1, +2] (SUPPORTS). D = 0 holds in only 5 of 8 models, and jlens_cascade/summary.txt gives the cross-model verdict as "mixed". *Correction:* report onsets as coinciding in 5 of 8 models, with one model each way (OLMo knowledge first by 5 layers, Gemma-2-9B familiarity first by 1), following the "mixed" wording in jlens_cascade/summary.txt.
- **C-3 (items 33, 103): "these models rarely abstain" / "the other three families rarely say IDK."** "Rarely" holds only for the strict both-judges rule (0.05, 0.05, 0.10 in the obscure-unknown cell). Under DONT_KNOW or CLARIFY it is 0.32, 0.22 and 0.25. hedge_judge/summary.txt gives single-judge corrected rates of 0.76 (Qwen), 0.42 (Gemma) and 0.74 (Mistral) in that cell. The files disagree, and the scope word depends on the measure chosen. *Correction:* tie the statement to the primary measure ("under the both-judges rule, at most 0.10 in the obscure-unknown cell") and give the secondary values from hedge_measure_v2/summary.txt beside it. The κ failure on its own is enough to justify reporting Llama only.

### Other mismatches (do not change a conclusion)

- **M-1 (item 23), selected value.** The forced-answer log-odds defence quantifies only Gemma (pop +1.79, know -0.32). For Llama, effects.csv gives pop - know = -0.12 [-0.72, +0.45], and for Qwen -2.49 [-3.41, -1.64]. The popularity effect is positive in all four, but it exceeds knowledge only in Gemma. *Correction:* report the Llama pop - know difference from effects.csv hedge_logodds, or say the popularity effect is positive in all four.
- **M-2 (items 51, 105, 107), count.** The text says "three other measurements", the caption says "four measurements" and the Limitations say "four measurements agree in sign", but Table A has five rows (primary + 4), all positive. *Correction:* "four other measurements" / "five measurements".
- **M-3 (item 101), caption vs content.** The range 0.45-0.74 is cited to Fig 3b, but 0.74 is Qwen2.5-7B-Instruct, which Fig 3 does not plot. Base-model onsets span 0.45-0.64. *Correction:* cite outcomes.csv for the 8-model range, or quote 0.45-0.64 for the plotted base models.
- **M-4 (item 84), configuration.** Two never-abstained directions exist: d_nohedge_U (+0.041 over random, the one quoted and plotted) and d_nohedge_both (+0.061). The text does not say that the quoted version differs in construction from the main direction. *Correction:* name the variant and give the other one's value from direction_nohedge/llama/summary.txt.
- **M-5 (item 56), scope.** "full Llama pool" refers to the 2,303 labelled items. The pool itself has 2,673. *Correction:* "over all 2,303 labelled Llama questions".
- **M-6 (item 106), scope.** "7-9B models", but the lens study includes Gemma-2-2B and 2B-it. *Correction:* "2-9B models".
- **M-7 (item 89), wording.** "every layer": curves.csv stops at layer n_layers - 2. *Correction:* "every layer but the last".
- **M-8 (items 127, 139), criterion compliance.** The hedge_measure_v2 criterion requires the regex results for the low-κ models to be reported with their recall (0.58/0.48/0.73), and the lens criterion requires base and instruct midF side by side. Neither appears, and App D refers to an instruct-versus-base comparison that the paper never shows. *Correction:* add the recall row to Table B and the instruct midF values (outcomes.csv), or remove the App D reference.
- **M-9 (file disagreement, item 16).** Llama's labelled set is 2,303 (763 HK / 1,540 U) in factorial_4models and unireps_checks, but 2,271 (758 / 1,513) of 2,637 candidates in data/familiarity_v3/gate_report.json. The lens criterion also says all factorials share 2,637 candidates, while the Llama factorial has 2,673. The paper's 2,303 matches the analysis files; the gate report looks stale.

### No-evidence-found (8)
Items 35 (10,000 bootstraps for the non-hedge intervals), 36 and 137 (criterion hashes), 63 (individual random-direction dots not recorded in provenance), 131 (entropy neurons as word-frequency/temperature neurons), 136 (the "behavioural bar fixed in advance" for injection), 138 (criteria for the four-family design and the never-abstained control not in the permitted files), 140 (citation counts).

### Notes
- "At least four times" (item 55) holds on point estimates (Llama 4.17) but not at the Llama CI bounds (3.37).
- "About a quarter" (item 50) = 0.23.
- The circuit recovery value 0.83 is the control-to-uncertain direction; uncertain-to-control is 1.03.
- The injection flip rate of 0.40 rises to 0.48 under the amended detector.
- The hedge_measure_v2 criterion file's mtime equals the item-sampling files' mtime (Oct 7 00:48), so "fixed before the run" cannot be confirmed from dates.
