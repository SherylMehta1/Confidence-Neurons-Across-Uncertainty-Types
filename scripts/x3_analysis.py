"""X3 analysis: the hashed scripts/x2_analysis.py run unchanged except for its input and output folders and the model name in the verdict text."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
src = (REPO / "scripts/x2_analysis.py").read_text(encoding="utf-8")
for a, b in (('D, OUT = REPO / "results/x2/raw", REPO / "results/x2"', 'D, OUT, SRC = REPO / "results/x3/raw", REPO / "results/x3", REPO / "results/x2"'), ('OUT / "sample.jsonl"', 'SRC / "sample.jsonl"'), ('say(f"X2: ', 'say(f"X3 (Gemma-2-9B-it, permissive prompt): '),
             ("the popularity effect holds across the range (Llama, PopQA)", "the effect replicates on Gemma-2-9B-it (PopQA)"), ("report as a limit of the extreme-tenths result", "not replicated on Gemma-2-9B-it; claims stay named to Llama")):
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, str(REPO / "scripts/x2_analysis.py"), "exec"), {"__file__": str(REPO / "scripts/x2_analysis.py"), "__name__": "__main__"})
