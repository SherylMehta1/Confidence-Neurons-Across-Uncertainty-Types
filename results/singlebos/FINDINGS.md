# Paper Figures 4 and 5 regenerated with a single BOS (7 Oct 2026)

- **Why:** the committed lens analyses passed chat prompts that already begin with BOS text to `jlens`, which adds its own BOS, so the model saw two (`results/jlens_bos_audit/FINDINGS.md`).
- **Second effect of the bug:** `jlens_positions.py` computes entity positions from single-BOS tokens but indexed `jlens` output that had one extra leading position, so the committed entity-span curve decoded the token **before** each entity token.
- **What was run:** the same three scripts, now with an opt-in `--strip-bos` flag (default off, so the committed results still reproduce), on the same pairs (195 / 234 / 178) with the same lenses and arguments. Outputs are in `results/singlebos/`. The figures were redrawn by `scripts/make_fig45_singlebos.py`, which reuses `make_paper_figures.py`'s fig4/fig5 code with only the input paths and output folder changed; they are in `paper/figures_singlebos/`. The committed figures are untouched.
- **Qwen is unaffected:** its tokenizer has no BOS token, and its numbers reproduce exactly.

## Every lens number the paper quotes

| Paper text (`paper/main.tex`) | Committed | Single BOS | Status |
|---|---|---|---|
| Llama familiarity: hedge log-odds gap half its final size by **L16**; entropy gap not until **L28** | L16 / L28 | L16 / L28 | unchanged |
| Qwen contested: by **L21** | L21 | L21 | unchanged (no BOS) |
| " unknown" at the uncertain readout: rank "~1,000" (L13) → "10" (L17) | 1,098 → 10 | **1,480 → 6.5** | update |
| "(single digits from L21)" | single digits L19-L30 | single digits at L17, L21-23, L25-28; 10-15 at L24, L29-30 | update |
| Control readout "never falls below ~300 (medians ~300-6,400 across L13-30)" | 308-6,387 | **512-15,024** | update |
| "both entity spans stay in the tens of thousands" | 10,821-42,593 | **16,659-101,244** | holds ("tens of thousands to ~100,000") |
| Steering at L15: "monotonic dose-response, ordered at every downstream layer, with margins narrowing above L26" | ordered at all layers | **not ordered at L27-28** (the α = −3 and α = 0 curves meet) | update |
| Decoded direction: " unknown" in top 10 at L15, top token from L17; contested vocabulary "both/either" → "conflicting/contradictory" | as stated | identical | unchanged |
| Llama contested lens log-odds 50% layer (not quoted) | L15 | L14 | n/a |

## Suggested replacement sentences

- **Abstract/intro:** "`` unknown'' rises from rank ${\sim}1{,}500$ to rank ${\sim}6$ across layers 13--17". The 12-layer lead over the entropy gap is unchanged.
- **Figure 5 caption:** "median lens rank of `` unknown'' at the uncertain readout falls two orders of magnitude, from ${\sim}1{,}500$ at L13 to ${\sim}6$ at L17 (single digits at most layers from L21), while the control readout never falls below ${\sim}500$ (medians ${\sim}500$--$15{,}000$ across L13--30) and both entity spans stay between ${\sim}17{,}000$ and ${\sim}100{,}000$. Right: steering the direction at L15 on known twins shifts the rank in order of dose at every downstream layer up to L26; at L27--28 the $-3\sigma$ and $0$ curves meet."
- **Methods** (one sentence): "Lens inputs use a single BOS token; an earlier version with a doubled BOS gave the same onset layers (appendix)."
