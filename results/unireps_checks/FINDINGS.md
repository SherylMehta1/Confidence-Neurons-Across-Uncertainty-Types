# Pre-submission confound checks (5 Oct 2026)

Script: `scripts/unireps_confound_checks.py` (CPU only; needs `wordfreq` and a Llama-3.1 `tokenizer.json`). Full numbers: `summary.txt` and the CSVs in this folder. All results are Llama-3.1-8B-Instruct, from committed data; nothing was re-generated.

## Check 1: is the effect about the entity, or about how its name is spelled?

**The popularity effect survives, but name spelling explains part of the headline contrast.** Controlling surface form shrinks the `pop_given_known` hedge gap by about 40% (+0.76 to +0.43); excluding name-inferable answers sharpens the dissociation (entropy gap +0.29 to +0.09).

| Test | Result |
|---|---|
| Full scored pool (2,303 items, 16 relations, relation fixed effects) | Popularity predicts hedging at -1.87 logits per SD. Adding name word frequency, token count and token rarity: **-1.79** (p < 1e-15). Surface form adds a little on its own (rarest word in the name: -0.72 per SD, p = 0.014) |
| `pop_given_unknown` pairs (n = 361) | Hedge gap +0.578 raw; **+0.572** at equal surface form |
| `pop_given_known` pairs (n = 55, the headline contrast) | Hedge gap +0.758 raw; **+0.434 [+0.18, +0.69]** at equal surface form, so surface form explains about 40% of this contrast |
| Name-inferable answers ("capital of Belorechensky District" → Belorechensk) | 35% of obscure-known twins in `pop_given_known`. Without them: hedge gap **+0.79 [+0.66, +0.90]**, entropy gap falls from +0.29 to **+0.09** |

**Correction to the headline wording.** Knowledge is not irrelevant to hedging. Raw cell means:

| | Known | Unknown |
|---|---|---|
| Popular | hedge 0.025, entropy 0.86 | hedge 0.236, entropy 2.58 |
| Obscure | hedge 0.678, entropy 0.95 | hedge 0.849, entropy 3.51 |

Popularity moves hedging by about 0.6 and knowledge by about 0.2. For entropy it is roughly reversed: knowledge moves it by 1.7-2.6 nats and popularity by 0.1-0.9. So the supportable claim is **"hedging tracks familiarity far more than knowledge; entropy tracks knowledge far more than familiarity"**, not "hedging ignores knowledge". A title saying "not when the fact is unknown" overstates this.

## Check 2: why does injection fail on facts the model knows? (T1)

**Mostly distance from the tipping point, not a separate "knowledge veto". The first-token hedge score overstates control over generation.**

- **Within the familiarity arm** (all targets known, 103 pairs), injection is less likely to work when the model is more confident in its answer. Target top-1 probability: 0.68 when injection works, 0.84 when it fails (p = 0.0006). The same holds for entropy (p = 0.0004).
- **Across the four arms**, the knowledge effect (-1.28, p < 1e-4) shrinks to -0.35 (p = 0.42) once the target's answer confidence is controlled. Known and unknown targets differ mainly in confidence: top-1 probability 0.77 vs 0.38. With only four arms this cannot fully separate the two accounts, but it does not favour a two-input gate over a margin effect.
- **The first-token hedge-vs-answer log-odds transfers almost fully even when generation does not flip.** In the familiarity arm the median recovery is 0.99 for failures and 1.04 for successes (0.90 and 0.93 for failures in the two factorial arms). 77% of failed known-target injections do not open with a hedge phrase, and about two-thirds of all failures restate the model's own unpatched answer (e.g. "Stephen King."); a few give other answers. The model's specific answer outcompetes the shifted hedge-vs-answer readout.
- **Consequence for the paper.** The 0.78-1.07 "recovery" numbers are first-token log-odds. They measure the hedge-vs-answer readout, not generation. Say so next to them, and lead the causal claim with the generation test.

**Suggested wording:** *"Injecting the circuit's state moves the first-token hedge score completely, but the model still answers when it is confident in a specific answer: injection succeeds less often the higher the target's answer probability (p < 0.001), which accounts for the apparent dependence on knowledge."*

## Check 3: a gap in the hedge detector (sensitivity only)

`behavioral_test.HEDGE_RE` matches "not available" but not "**not readily available** in my database" or "not immediately available", so those outputs were scored as answers. The original detector reproduces the stored labels exactly (1.000).

| Arm | Injection, original | Injection, amended |
|---|---|---|
| familiarity_v3 (pre-committed test) | 0.398 [0.301, 0.495] | **0.481 [0.380, 0.574]** |
| know_given_hi | 0.214 | 0.286 |
| pop_given_unknown | 0.653 | 0.689 |

Removal is unchanged (0.98 → 0.97). **The pre-committed verdict stands (EQUIVOCAL):** the locked analysis used the original detector, and the amended point estimate is still below 0.50. But the commit message's sentence "the CI tops out at 0.495, below the 0.50 floor" is detector-dependent; with the amended detector the CI includes 0.50. Report the locked result as primary and this as a disclosed sensitivity analysis. Do not swap the primary.


*Corrected 7 Oct 2026 after an independent audit of every number in this file: wording and figures fixed where the audit found them wrong or overstated.*
