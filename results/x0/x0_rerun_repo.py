"""Re-run the repo's own analysis scripts on the raw files, with every write redirected to OUT/repo_reruns (tracked files untouched),
and diff the text against the committed outputs."""
import difflib
import runpy
import shutil
import tempfile
from x0_common import *

DST = OUT / "repo_reruns"; DST.mkdir(exist_ok=True)
log = Log("x0_rerun_log.txt")
norm = lambda s: re.sub(r"np\.int64\((\d+)\)", r"\1", s)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    fake = tmp / "repo"; (fake / "paper_2.0/analysis").mkdir(parents=True); (fake / "scripts").mkdir()
    (fake / "data").symlink_to(REPO / "data")
    shutil.copytree(REPO / "results/hedge_measure_v2", fake / "results/hedge_measure_v2")
    shutil.copytree(REPO / "results/hedge_judge", fake / "results/hedge_judge")
    (fake / "results/round3").symlink_to(REPO / "results/round3")
    (fake / "Confidence-Neurons-Across-Uncertainty-Types").symlink_to(fake)  # name some scripts hard-code
    shutil.copy(REPO / "scripts/behavioral_test.py", fake / "scripts")
    for rel, out in (("scripts/hedge_measure_v2_analysis.py", "results/hedge_measure_v2/summary.txt"), ("scripts/hedge_judge_analysis.py", "results/hedge_judge/summary.txt"),
                     *[(f"paper_2.0/analysis/{n}.py", f"paper_2.0/analysis/{n}.txt") for n in ("echo_check", "round2_checks", "round3_label_checks", "round4_checks", "within_relation_check", "mere_checks", "lockin_checks")]):
        shutil.copy(REPO / rel, fake / rel)
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(fake / rel), run_name="__main__")
            new, old = (fake / out).read_text(encoding="utf-8"), (REPO / out).read_text(encoding="utf-8")
            (DST / Path(out).name if "summary" not in out else DST / (Path(rel).stem + "_summary.txt")).write_text(new, encoding="utf-8")
            d = [l for l in difflib.unified_diff(norm(old).splitlines(), norm(new).splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
            log(f"{rel}: " + ("identical to the committed output" if not d else f"{len(d)} differing lines"))
            for l in d:
                log("    " + l[:260])
        except Exception as e:
            log(f"{rel}: did not run here ({type(e).__name__}: {e})")
log.save()
