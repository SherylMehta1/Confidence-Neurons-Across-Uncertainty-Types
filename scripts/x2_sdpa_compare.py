"""X2 amendment: run the hashed scripts/x2_analysis.py unchanged except that Gemma's labels come from the paper's sdpa path.
Outputs in results/x2/paper_sdpa_judge/."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
src = (REPO / "scripts/x2_analysis.py").read_text(encoding="utf-8")
new = src.replace('D / f"judge_{j}_full.jsonl"', 'D / ("judge_gemma27b_sdpa_full.jsonl" if j == "gemma27b" else f"judge_{j}_full.jsonl")')
assert new != src
n2 = new.replace('OUT / "sample.jsonl"', 'SRC / "sample.jsonl"').replace('REPO / "results/x2"\n', 'REPO / "results/x2/paper_sdpa_judge"; SRC = REPO / "results/x2"; OUT.mkdir(exist_ok=True)\n')
assert n2 != new
exec(compile(n2, str(REPO / "scripts/x2_analysis.py"), "exec"), {"__file__": str(REPO / "scripts/x2_analysis.py"), "__name__": "__main__"})
