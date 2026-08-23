import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_audit as ja


def test_items_strata_and_summary(tmp_path):
    d = tmp_path / "fam"; d.mkdir()
    recs = [dict(prompt_id=f"u{i}", raw_prompt=f"Who wrote Book{i}?", gold=["Ann Lee", "A. Lee"], greedy="Ann Lee" if i % 2 else "Bob", samples=["ann lee", "unknown"], hedge_rate=0.67 if i % 2 else 0.0, slick_class="Unknown") for i in range(6)]
    with open(d / "prompts.jsonl", "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    with open(d / "controls.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(prompt_id="c0", raw_prompt="Capital of France?", gold=["Paris"], greedy="Paris.", samples=["paris", "Lyon"], hedge_rate=0.0)) + "\n")
    items = ja.load_items(tmp_path, "fam")
    assert len(items) == 6 * 3 + 3 and {i["side"] for i in items} == {"uncertain", "control"}
    assert [i for i in items if i["prompt_id"] == "c0" and i["kind"] == "greedy"][0]["grader"] is True
    sample, sizes = ja.stratified_sample(items, 2, 0)
    assert all(v <= 2 for v in [sum(1 for s in sample if (s["grader"], s["side"], s["hedged"]) == eval(k)) for k in sizes])
    df = pd.DataFrame(items); df["gold"] = df.gold.apply(" | ".join); df["judge"] = df.grader; df.loc[df.index[:3], "judge"] = ~df.grader[:3]
    df["agree"] = df.grader == df.judge; df["kind_group"] = df.kind.where(df.kind == "greedy", "sample")
    s = ja.summarize(df, "fam", "judge-x", sizes)
    assert "overall agreement" in s and "grader false negatives" in s and "hedged" in s
    assert abs(ja.kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1.0) < 1e-9 and abs(ja.kappa([1, 0, 1, 0], [1, 1, 0, 0])) < 1e-9
