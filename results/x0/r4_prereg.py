"""R4: pre-registration table. One row per plan in paper_2.0/PREREG, then a separate block for root-level criterion files.
Hashes are sha256 of the file bytes, first 8 characters. Git dates are committer-side author dates of this repository."""
import hashlib
import subprocess
from x0_common import *

git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip().splitlines()
# only the analysis' own result files: text and markdown under results/, minus the x0 folder
RES = [p for p in sorted((REPO / "results").rglob("*")) if p.suffix in (".txt", ".md") and "x0" not in p.relative_to(REPO).parts and p.is_file()]
TXT = {p: p.read_text(encoding="utf-8", errors="replace").splitlines() for p in RES}
OTHER = {p: p.read_text(encoding="utf-8").splitlines() for p in (REPO / "paper_2.0/PREREG/HASHES_round3.txt", REPO / "paper_2.0/LAB_LOG.md")}
H8 = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")


def first_last(path):
    log = git("log", "--follow", "--format=%h|%aI", "--", path)
    return len(log), log[-1].split("|") if log else ("", ""), log[0].split("|") if log else ("", "")


def quotes(name, h):
    """Lines in result files that name the plan (file stem) next to a hash, or quote the current hash."""
    hits = []
    for p, ls in TXT.items():
        for i, l in enumerate(ls, 1):
            near = set(H8.findall(l))
            if h in near:
                hits.append((p, i, h))
            elif name in l and near:  # plan named next to another hash only
                hits.append((p, i, sorted(near)[0]))
    return hits


NOTE = {"cross_model_agreement.md": "its own text says it was written after the point estimates had been printed", "e17_nla_qwen.md": "exploratory; the run's output was lost (LAB_LOG), so no result file exists",
        "e4_amendment_third_judge.md": "amendment to E4; secondary scorer only"}
# result file(s) whose first commit is compared with the plan's first commit
RESFILE = {"e1_": "e1", "e2_": "e2", "e3_": "e3", "e4_": "e4", "e5_": "e5", "e6_": "e6", "e7_": "e7", "e8_": "e8", "e9_": "e9", "e10_": "e10", "e11_": "e11", "e12_": "e12", "e13_": "e13", "e15_": "e15"}
ROOT_RES = {"direction_reuse_criterion.md": "results/direction_reuse/FINDINGS.md", "direction_reuse_v2_criterion.md": "results/direction_reuse_v2/FINDINGS.md", "gemma_mistral_factorial_criterion.md": "results/factorial_4models/FINDINGS.md",
            "hedge_judge_criterion.md": "results/hedge_judge/summary.txt", "hedge_measure_v2_criterion.md": "results/hedge_measure_v2/summary.txt", "jlens_cascade_criterion.md": "results/jlens_cascade/FINDINGS.md",
            "l5_behavioral_criterion.md": "results/circuit_familiarity_v3/behavioral_result.txt", "preregistration_commit.md": "results/circuit_uu_hedge_confab_v3/step_f_confirmatory_result.txt", "nohedge_direction_criterion.md": "results/direction_nohedge/FINDINGS.md", "qwen_factorial_criterion.md": "results/qwen_factorial/FINDINGS.md"}
rows = []
plans = [(p, "PREREG folder") for p in sorted((REPO / "paper_2.0/PREREG").glob("*.md"))] + [(p, "root criterion files") for p in sorted(REPO.glob("*_criterion.md")) + [REPO / "preregistration_commit.md"]]
for p, block in plans:
    rel = p.relative_to(REPO).as_posix()
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    q = quotes(p.name, h)
    good = [x for x in q if x[2] == h]
    where = lambda xs: "; ".join(f"{a.relative_to(REPO).as_posix()}:{i}" for a, i, _ in xs)
    rec = [f"{a.relative_to(REPO).as_posix()}:{i}" for a, ls in OTHER.items() for i, l in enumerate(ls, 1) if h in H8.findall(l)]
    n, first, last = first_last(rel)
    if block == "PREREG folder":
        tag = next((v for k, v in RESFILE.items() if p.name.startswith(k)), None)
        rf = f"results/round3/{tag}_summary.txt" if tag else ("paper_2.0/analysis/cross_model_agreement.txt" if p.name.startswith("cross") else "")
    else:
        rf = ROOT_RES.get(p.name, "")
    rn, rfirst, _ = first_last(rf) if rf else (0, ("", ""), None)
    if not rfirst[1]:
        order = "no result file"
    else:
        order = "plan first" if pd.Timestamp(first[1]) < pd.Timestamp(rfirst[1]) else ("result first" if pd.Timestamp(first[1]) > pd.Timestamp(rfirst[1]) else "same commit")
    wrong = sorted({x[2] for x in q if x[2] != h})
    if good:
        status = "hash quoted in results"
    elif rec:
        status = "hash recorded outside the results only"
    elif block == "PREREG folder":
        status = "post hoc (no recorded hash)"
    else:
        status = "no recorded hash"
    rows.append(dict(block=block, file=rel, sha256_8=h, hash_quoted=h if good else "", quoted_in_results=where(good), recorded_elsewhere="; ".join(rec), match=True if good else ("no quoted hash" if not wrong else False),
                     other_hash_on_a_line_naming_the_plan="; ".join(f"{x[2]} at {x[0].relative_to(REPO).as_posix()}:{x[1]}" for x in q if x[2] != h), git_commits=n, first_commit=first[0], first_commit_date=first[1],
                     last_edit_commit=last[0], last_edit_date=last[1], edits_after_first_commit=n - 1, result_file=rf, result_first_commit=rfirst[0], result_first_commit_date=rfirst[1], git_order=order, status=status,
                     note=NOTE.get(p.name, "")))
T = pd.DataFrame(rows); save(T, "prereg_table.csv")
print(T[["file", "sha256_8", "match", "git_commits", "first_commit_date", "result_first_commit_date", "git_order", "status"]].to_string(index=False))
for b, g in T.groupby("block", sort=False):
    print(b, len(g), "plans;", (g.match == True).sum(), "match a hash quoted in the results;", (g.edits_after_first_commit > 0).sum(), "edited after first commit; git order:", g.git_order.value_counts().to_dict())
x = T[T.other_hash_on_a_line_naming_the_plan != ""][["file", "other_hash_on_a_line_naming_the_plan"]]
print("other hashes on lines naming a plan:\n", x.to_string(index=False) if len(x) else "none")
assert (T.block == "PREREG folder").sum() == 17
