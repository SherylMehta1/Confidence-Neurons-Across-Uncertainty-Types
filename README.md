# Do All Roads Lead to the Same Neurons?

Testing whether confidence-regulating neurons in Llama-3.1-8B-Instruct generalize across
three distinct uncertainty types: **ambiguity**, **lack of knowledge**, and **contradictory context**.

## Team structure

- **Person A** — ambiguity (`person_A_ambiguity/`), data source: AmbigQA/AmbigNQ
- **Person B** — lack of knowledge (`person_B_lack_of_knowledge/`), data source: UnknownBench (NEC subset)
- **Person C** — contradictory context (`person_C_contradictory_context/`), data source: Tighidet et al. knowledge-probing framework

## How this repo is organized

```
shared/                 <- built together, Weeks 1-2. Nobody forks this privately.
  model_utils.py           Tools 1-2: load model, run forward pass, compute entropy
  detection.py              Tool 3 + Phase 2: neuron detection (correlation scan)
  ablation.py                Tool 4: mean-ablation (Phase 4 causal test)
  logit_lens.py               Tool 5: direct-effect decomposition (Phase 3 mechanism check)

data/                   <- category datasets, pulled from the three source repos (see DATA_SOURCES.md)
  ambiguity/
  lack_of_knowledge/
  contradictory_context/

candidate_neurons.json  <- SHARED, FIXED output of Phase 2. Produced once, read by everyone. Do not edit by hand.

person_A_ambiguity/          <- individual work, Phase 1/3/4, using ONLY shared/ functions
person_B_lack_of_knowledge/
person_C_contradictory_context/
  notebooks/
  results/                     <- each person's results CSV, common schema (see RESULTS_SCHEMA.md)

results/                <- merged, team-level results (Week 9-10 synthesis)
notebooks/              <- shared exploratory / hello-world notebooks
```

## Ground rules

1. **`shared/` is built together and reviewed by all three before anyone builds on it.**
   If you need to change a shared function, change it in `shared/`, tell the team, don't fork a private copy.
2. **`candidate_neurons.json` is produced once (Phase 2, done together) and then frozen.**
   Nobody hand-picks their own neurons after this point.
3. **Every person's results CSV uses the exact schema in `RESULTS_SCHEMA.md`.**
   This is what makes the Week 9 merge possible without a data-wrangling nightmare.
4. **Commit early, commit often.** Kaggle/Colab sessions can disconnect — don't lose work.

## Setup

See `SETUP.md` for environment setup (Kaggle, HuggingFace access, dependencies).

## Papers this project builds on

See `REFERENCES.md`.
