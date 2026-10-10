"""
Redraw paper Figures 4 and 5 from the single-BOS lens reruns (results/singlebos/, made with --strip-bos; see
results/jlens_bos_audit/FINDINGS.md). Reuses make_paper_figures.py's palette, helpers and the exact fig4/fig5 code,
changing only the input paths and the output folder (paper/figures_singlebos/). The committed figures are untouched.
"""
from pathlib import Path

src = Path(__file__).with_name("make_paper_figures.py").read_text(encoding="utf-8")
pre = src[: src.index("# ---- fig1")]
blk = src[src.index("# ---- fig4"): src.index("# ---- fig6")]
for a, b in [('"paper/figures"', '"paper/figures_singlebos"'), ('f"paper/figures/{name}.pdf"', 'f"paper/figures_singlebos/{name}.pdf"'),
             ('f"figures/{name}.svg"', 'f"paper/figures_singlebos/{name}.svg"')]:
    assert pre.count(a) == 1, a
    pre = pre.replace(a, b)
for a, b in [("results/circuit_familiarity/jlens_trajectory.csv.gz", "results/singlebos/circuit_familiarity/jlens_trajectory.csv"),
             ("results/circuit_conflict/jlens_trajectory.csv.gz", "results/singlebos/circuit_conflict/jlens_trajectory.csv"),
             ("results/circuit_conflict_qwen/jlens_trajectory.csv.gz", "results/singlebos/circuit_conflict_qwen/jlens_trajectory.csv"),
             ("results/circuit_familiarity/jlens_entity.csv", "results/singlebos/circuit_familiarity/jlens_entity.csv"),
             ("results/circuit_familiarity/jlens_steer.csv", "results/singlebos/circuit_familiarity/jlens_steer.csv")]:
    assert blk.count(a) == 1, a
    blk = blk.replace(a, b)
exec(compile(pre + "\n" + blk, "make_fig45_singlebos", "exec"))
