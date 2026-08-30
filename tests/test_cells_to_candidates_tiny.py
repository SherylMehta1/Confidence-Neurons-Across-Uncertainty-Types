import json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_adapter_pairs_within_relation(tmp_path):
    low = tmp_path / "R2.jsonl"; high = tmp_path / "R1.csv"
    with open(low, "w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps(dict(id=f"l{i}", question=f"Who is the author of Book{i}", possible_answers=["X", "Y"], prop="author", s_pop=10 + i, split="train")) + "\n")
        f.write(json.dumps(dict(id="l9", question="What genre is Film9", possible_answers="['drama']", prop="genre", s_pop=3)) + "\n")
    high.write_text("id,question,answers,prop,s_pop,split\n" + "".join(f"h{i},Who is the author of Famous{i},A|B,author,9000,test\n" for i in range(4)) + "h9,What genre is Hit9,comedy,genre,8000,train\n", encoding="utf-8")
    cat = "tinygrid_test"
    r = subprocess.run([sys.executable, str(REPO / "scripts/cells_to_candidates.py"), "--low", str(low), "--high", str(high), "--category", cat], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = REPO / "data" / cat / "candidates.jsonl"
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    try:
        assert len(rows) == 5  # 4 author + 1 genre controls available
        assert all(x["uncertain"]["meta"]["relation"] == x["control"]["meta"]["relation"] for x in rows)
        g = [x for x in rows if x["uncertain"]["meta"]["relation"] == "genre"][0]
        assert g["uncertain"]["gold"] == ["drama"] and g["control"]["gold"] == ["comedy"]
        assert rows[0]["control"]["gold"] == ["A", "B"] and rows[0]["uncertain"]["question"].endswith("?")
        assert "2 low items unmatched" in r.stdout
    finally:
        out.unlink(); out.parent.rmdir()
