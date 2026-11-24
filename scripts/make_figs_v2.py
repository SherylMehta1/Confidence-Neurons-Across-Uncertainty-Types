"""Figures for the v2 drafts: the paper's own figure scripts, run on the corrected labels in a shadow repo (scripts/rejudge_shadow.py DIR corrected),
with two layout changes (legend position in 1a; y-axis of the steering panel, which clipped). Writes paper_2.0/figures/fig_{dissociation,steering}_v2.*
  python scripts/make_figs_v2.py DIR"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO, SH = Path(__file__).resolve().parents[1], Path(sys.argv[1]).resolve()
F = SH / "paper_2.0/figures"
for name, subs in (("fig_dissociation", [(r"legend\((.*?)\)", None)]), ("fig_steering", [("axs[0].set_ylim(0, 0.25)", "axs[0].set_ylim(0, 0.6)")])):
    s = (REPO / f"paper_2.0/figures/{name}.py").read_text(encoding="utf-8")
    if name == "fig_dissociation":
        n = re.sub(r"axs\[0\]\.legend\([^\n]*\)", 'axs[0].legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.0, labelspacing=0.2)', s)
        n = n.replace('interaction(axs[0], ca, "share abstaining", (0, 1))', 'interaction(axs[0], ca, "share abstaining", (0, 1.25)); axs[0].set_yticks([0, 0.5, 1])')
    else:
        n = s.replace(*subs[0])
    assert n != s, name
    n = n.replace(f'"{name}"', f'"{name}_v2"')
    (F / f"{name}_v2.py").write_text(n, encoding="utf-8")
    subprocess.run([sys.executable, f"{name}_v2.py"], cwd=F, check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONWARNINGS="ignore"), stdout=subprocess.DEVNULL)
    for e in ("pdf", "png"):
        shutil.copy(F / f"{name}_v2.{e}", REPO / "paper_2.0/figures")
    print(name, "->", f"paper_2.0/figures/{name}_v2.pdf")
