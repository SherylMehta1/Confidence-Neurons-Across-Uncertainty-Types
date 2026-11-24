# Regenerating Final Numbers and Figures Before Submission

*Confidence Neurons Across Uncertainty Types — UniReps 2026 resubmission*

## Why This Is Necessary

Across this entire drafting process, numbers have been copied by hand from result files into paragraphs, checked one at a time, and corrected multiple times (a saturation claim that referenced a non-existent N=40 file, a bridge/alignment paragraph that initially mixed up which pool each of three runs actually used, an "effective dimensionality" figure that does not exist anywhere in the repository, a 61% relation-share figure that should have been 59%). Every one of these was caught only because each paragraph was checked individually against the underlying files. That approach does not scale to a full paper, and it is exactly the failure mode the project's own tooling exists to prevent.

The repository already contains two scripts built for this: `scripts/make_paper_figures.py`, which renders every paper figure directly from committed result CSVs, and `scripts/make_numbers_tex.py`, which writes one LaTeX macro per citable number, sourced from those same committed files or from a figure's own cell data. Once these are run and `main.tex` is updated to cite the generated macros instead of hand-typed numbers, a number can no longer silently drift from the result that produced it.

## Before Running Anything

- Confirm every experiment that the paper cites has actually finished and its output files are committed to the `l0-fixes` branch. This step cannot be done from this verification session — it requires checking with whoever is running the GPU jobs.
- Pull the latest `l0-fixes` branch fresh immediately before running either script, so nothing stale is read.

## Step-by-Step

1. **Regenerate every figure first:**

   ```bash
   python scripts/make_paper_figures.py
   ```

   This produces `paper/figures/fig1.pdf` through `fig8.pdf`, and writes each figure's underlying per-cell data (for example `paper/figures/fig6_familiarity_v3_cells.csv`) back to disk. These per-cell files matter beyond the figures themselves — the numbers script reads confidence intervals from them in preference to recomputing its own, specifically so a number quoted in prose always matches the figure a reader can see next to it.

2. **Regenerate the numbers file:**

   ```bash
   python scripts/make_numbers_tex.py
   ```

   This overwrites `paper/numbers.tex` with one `\newcommand` per citable number, each one computed directly from a committed result file (never typed by hand), and each one's source file recorded alongside it in the script's own report. It performs no analysis of its own — it only aggregates what the experiment scripts already produced.

3. **Run the check mode before trusting the output:**

   ```bash
   python scripts/make_numbers_tex.py --check
   ```

   This exits with an error and writes nothing if any file the manuscript depends on is missing, rather than silently leaving a gap. Run this and resolve anything it flags before moving on.

4. **Read the script's own printed report carefully.**

   The script deliberately does not pick a single "canonical" circuit on its own. Every dataset draw is emitted under its own separate suffix: `Old` (the original pre-dedup pool), `New` (familiarity_v2), `VThree` (familiarity_v3 at the default 60 held-out pairs), and `VThreeFull` (familiarity_v3 at the full 141). A human has to decide which suffix's numbers actually belong in the main text — for this submission, that should be `VThree` or `VThreeFull`, never `Old` or `New`, since those are the superseded pools this paper no longer reports on.

5. **Update `main.tex` to cite the generated macros, not typed-in numbers.**

   Wherever the current draft has a hand-typed figure (for example "0.827"), replace it with the matching macro (for example `\circuitHedgeCUVThree`) once the macro's value has been confirmed to match. This is the step that actually closes the drift risk — regenerating the files does nothing if `main.tex` still quotes the old typed-in numbers next to them.

## One Discrepancy to Resolve Before Trusting the Alignment Number

While checking how the bridge/alignment macro is sourced, a real inconsistency turned up that should be fixed, not just noted. The script reads the headline alignment number (`alignCosLThirtyOne`) from `results/bridge_familiarity_v3/cos_bootstrap.csv` — but that specific file's own provenance record states `category: familiarity_v2`, meaning it was actually run on the earlier, superseded pool, not on familiarity_v3 despite the directory's name. The run that was actually built on familiarity_v3 lives in a differently-named directory, `results/bridge_familiarity_v5/cos_bootstrap.csv`.

This matters because this paper's scope is restricted to familiarity_v3 only. Before relying on the generated `alignCosLThirtyOne` macro, either point the script at the v5 directory's file, or confirm with whoever maintains this script which of the two was actually intended as the canonical familiarity_v3 alignment run — this is a judgment call about the repository's own internal naming, not something to silently resolve by picking one.

## After Regeneration

- Recompile the PDF once `numbers.tex` has been regenerated and `main.tex` updated to cite its macros, and visually check the rendered figures and numbers together, the same way earlier drafts in this process were checked page by page.
- Re-run this same two-script pipeline any time an upstream result changes, rather than hand-editing a single number in the draft — that hand-editing is exactly the practice this pipeline exists to replace.
- Decide, separately, what to do about the missing familiarity_v3 judge-audit run (grader-reliability check) — this is not something either script covers, since no such audit has ever been run on familiarity_v3. That is a separate experiment to run or a separate limitation to disclose, not a regeneration step.
