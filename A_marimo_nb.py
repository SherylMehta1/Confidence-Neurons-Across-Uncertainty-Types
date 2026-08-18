# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "shared==0.0.32",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    from huggingface_hub import whoami

    try:
        info = whoami()
        print("✅ Hugging Face token is working!")
        print("Logged in as:", info["name"])
    except Exception as e:
        print("❌ Hugging Face authentication failed")
        print(type(e).__name__, str(e))
    return


@app.cell
def _():
    CATEGORY = "ambiguity"   # or "ambiguity" / "contradictory_context"
    DATA_PATH = f"data/{CATEGORY}/prompts.jsonl"
    RESULTS_PATH = f"person_A_ambiguity/results/results.csv"  # adjust folder per person
    return CATEGORY, RESULTS_PATH


@app.cell
def _():
    from shared.detection import load_candidate_neurons
    candidates = load_candidate_neurons("candidate_neurons.json")
    print(f"{len(candidates)} candidates loaded")
    return (candidates,)


@app.cell
def _():
    #from shared.detection import load_candidate_neurons
    #candidates = load_candidate_neurons("candidate_neurons.json")
    #print(f"{len(candidates)} candidates loaded")

    baseline_prompts = [
        "The weather today is", "My favorite hobby is", "The history of Rome began",
        "She walked into the room and", "The stock market today", "In the morning, I usually",
        "The scientist explained that", "According to the report, the company",
        "The recipe calls for", "He picked up the phone and",
        "The movie was about", "Scientists recently discovered",
        "The government announced that", "On weekends, she likes to",
        "The book begins with", "After the meeting, they decided to",
        "The restaurant serves", "In ancient times, people believed",
        "The engineer designed a", "Every year, the festival", "The children played in the",
        "Traffic was heavy because",
        "The professor lectured about",
        "New technology allows people to",
        "The garden was full of",
        "Doctors recommend that patients",
        "The company's quarterly report showed",
        "During the summer, families often",
        "The artist painted a picture of",
        "The judge ruled that",
        "Farmers in the region grow",
        "The airline announced delays due to",
        "The museum's new exhibit features",
        "Local officials confirmed that",
        "The chef prepared a dish with",
        "Researchers are studying how",
        "The novel is set in",
        "Investors reacted to the news by",
        "The team's coach explained that",
        "Volunteers gathered to help with",
        "The bridge was built in",
        "Wildlife experts observed that",
    ]
    print(f"{len(baseline_prompts)} baseline prompts") 
    return (baseline_prompts,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(
    CATEGORY,
    baseline_prompts,
    candidates,
    held_out_prompts,
    model,
    pd,
    tokenizer,
    working_prompts,
):
    from shared.ablation import compute_mean_activation, run_ablation_experiment

    def run_full_category(prompts, split_name):
        rows = []
        for c in candidates:
            layer_idx, neuron_idx = c["layer"], c["neuron_idx"]
            mean_val = compute_mean_activation(model, tokenizer, baseline_prompts, layer_idx, neuron_idx)
            neuron_rows = run_ablation_experiment(
                model, tokenizer, prompts, layer_idx, neuron_idx, mean_val,
                category=CATEGORY, split=split_name
            )
            rows.extend(neuron_rows)
            avg_shift = sum(r["entropy_shift"] for r in neuron_rows) / len(neuron_rows)
            print(f"[{split_name}] {c['neuron_id']}: avg entropy_shift = {avg_shift:+.6f}")
        return rows

    working_rows = run_full_category(working_prompts, "working")
    held_out_rows = run_full_category(held_out_prompts, "held_out")

    all_rows = working_rows + held_out_rows
    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df)} ({len(working_rows)} working, {len(held_out_rows)} held_out)")
    return (df,)


@app.cell
def _(df):
    sample_entropy = df["orig_entropy"].iloc[0]
    print(f"Sample entropy: {sample_entropy:.6f}")
    print(f"e*256 mod 1 = {(sample_entropy*256) % 1:.6f}  <- should NOT be ~0")
    print(f"\nExact-zero shifts: {(df['entropy_shift'] == 0).sum()} / {len(df)} "
          f"({100*(df['entropy_shift']==0).mean():.1f}%)  <- should be near 0%, was 34-38% before the fix")
    return


@app.cell
def _(RESULTS_PATH, df):
    df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved {len(df)} rows to {RESULTS_PATH} (full replacement of stale pre-fix data)")
    return


@app.cell
def _(os, subprocess):
    gh_token = os.environ["github_access"]

    subprocess.run(["git", "config", "--global", "user.email", "your_email@example.com"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "your_github_username"], check=True)
    subprocess.run(
        ["git", "remote", "set-url", "origin",
         f"https://{gh_token}@github.com/SherylMehta1/Confidence-Neurons-Across-Uncertainty-Types.git"],
        check=True,
    )
    return


@app.cell
def _(subprocess):
    check_result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    print(check_result.stdout)
    return


@app.cell
def _(subprocess):
    # ------------------------------------------------------------
    # 2. Check what's changed BEFORE stashing (so you know what you're
    #    about to set aside)
    # ------------------------------------------------------------
    status_result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    print(status_result.stdout)
    return


@app.cell
def _(subprocess):
    # ------------------------------------------------------------
    # 3. Stash everything (including untracked new files, via -u),
    #    with a clear message
    # ------------------------------------------------------------
    stash_result = subprocess.run(
        ["git", "stash", "push", "-u", "-m", "Phase 1 fixes + ambiguity results.csv"],
        capture_output=True, text=True
    )
    print(stash_result.stdout, stash_result.stderr)
    return


@app.cell
def _(subprocess):
    # ------------------------------------------------------------
    # 4. Pull latest from teammates, now that your working tree is clean
    # ------------------------------------------------------------
    pull_result = subprocess.run(["git", "pull", "--rebase"], capture_output=True, text=True)
    print(pull_result.stdout, pull_result.stderr)
    return


@app.cell
def _(subprocess):
    # ------------------------------------------------------------
    # 5. Restore your stashed changes on top of the freshly-pulled code
    # ------------------------------------------------------------
    pop_result = subprocess.run(["git", "stash", "pop"], capture_output=True, text=True)
    print(pop_result.stdout, pop_result.stderr)
    return


@app.cell
def _(subprocess):
    # ------------------------------------------------------------
    # 6. Confirm stash restored correctly
    # ------------------------------------------------------------
    result2 = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    print(result2.stdout)
    return


@app.cell
def _(subprocess):
    # Step 7 -- stage results.csv
    add_result = subprocess.run(
        ["git", "add", "person_A_ambiguity/results/results.csv"],
        capture_output=True, text=True
    )
    print(add_result.stdout, add_result.stderr)

    staged_check = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    print(staged_check.stdout)
    return


@app.cell
def _(subprocess):
    # Step 8 -- commit
    commit_result = subprocess.run(
        ["git", "commit", "-m", "Phase 1 complete: fp32 entropy fix applied, results.csv regenerated"],
        capture_output=True, text=True
    )
    print(commit_result.stdout, commit_result.stderr)
    return


@app.cell
def _(subprocess):
    # Step 9 -- push
    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    print(push_result.stdout, push_result.stderr)
    return


@app.cell
def _():
    print("model_fn defined:", "model_fn" in dir())
    print("working_prompts_fn defined:", "working_prompts_fn" in dir())
    return


@app.cell
def _(model_fn, tokenizer_fn, working_prompts_fn):
    #sanity check for the hooks
    import torch as torch_snc
    from shared.model_utils import get_next_token_probs as get_next_token_probs_snc, compute_entropy as compute_entropy_snc

    def frozen_norm_no_ablation_snc(model_snc, tokenizer_snc, prompt_snc):
        inputs_snc = tokenizer_snc(prompt_snc, return_tensors="pt").to(model_snc.device)

        captured_snc = {}
        def capture_hook_snc(module, input, output):
            x_snc = input[0]
            variance_snc = x_snc.pow(2).mean(-1, keepdim=True)
            rms_snc = torch_snc.sqrt(variance_snc + module.variance_epsilon)
            captured_snc["rms"] = rms_snc.detach()

        norm_handle_snc = model_snc.model.norm.register_forward_hook(capture_hook_snc)
        with torch_snc.no_grad():
            _ = model_snc(**inputs_snc)
        norm_handle_snc.remove()

        def frozen_norm_hook_snc(module, input, output):
            x_snc = input[0]
            frozen_rms_snc = captured_snc["rms"]
            return x_snc / frozen_rms_snc * module.weight

        norm_freeze_handle_snc = model_snc.model.norm.register_forward_hook(frozen_norm_hook_snc)
        try:
            with torch_snc.no_grad():
                outputs_snc = model_snc(**inputs_snc)
        finally:
            norm_freeze_handle_snc.remove()

        logits_snc = outputs_snc.logits[0, -1, :]
        return torch_snc.nn.functional.softmax(logits_snc, dim=-1)


    # Use YOUR ambiguity model/tokenizer/prompts, already loaded from the
    # frozen_norm_ablate_fn script you just ran
    test_prompt_snc = working_prompts_fn[0]["chat_formatted_prompt"]

    orig_probs_snc = get_next_token_probs_snc(model_fn, tokenizer_fn, test_prompt_snc)
    orig_entropy_snc = compute_entropy_snc(orig_probs_snc)

    frozen_only_probs_snc = frozen_norm_no_ablation_snc(model_fn, tokenizer_fn, test_prompt_snc)
    frozen_only_entropy_snc = compute_entropy_snc(frozen_only_probs_snc)

    print(f"Original entropy: {orig_entropy_snc:.6f}")
    print(f"Frozen-norm-only entropy (no ablation): {frozen_only_entropy_snc:.6f}")
    print(f"Difference: {frozen_only_entropy_snc - orig_entropy_snc:.6f}")
    print("If this difference is NOT near-zero (e.g. > 0.001), the freeze-hook")
    print("mechanics are broken, and your ablation-included results shouldn't be trusted yet.")
    return


@app.cell
def _(model_fn):
    # Clear any stale/leftover hooks on model.norm from earlier failed attempts
    model_fn.model.norm._forward_hooks.clear()
    model_fn.model.norm._forward_pre_hooks.clear()
    print("Cleared stale hooks on model.norm")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(os, subprocess):
    # Step 7 — auth
    #import os
    hf_token = os.environ.get("HF_TOKEN")
    github_token = os.environ.get("github_access")

    from huggingface_hub import login
    login(token=hf_token)
    print("hi")
    subprocess.run(
        ["git", "config", "--global", "user.name", "kanikasinghal1612"],
        check=True
    )

    subprocess.run(
        ["git", "config", "--global", "user.email", "kanikasinghal1612@gmail.com"],
        check=True
    )
    print("GitHub token found:", github_token is not None)
    return


@app.cell
def _(importlib, os, sys):
    REPO_ROOT = "/marimo/Confidence-Neurons-Across-Uncertainty-Types"

    # 1. Force REPO_ROOT to position 0 in sys.path
    if REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

    # 2. Sync working directory
    os.chdir(REPO_ROOT)

    # 3. Clear Python's internal import cache
    importlib.invalidate_caches()
    return


@app.cell
def _():

    def _():
        import sys
        import subprocess
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "marimo",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "huggingface_hub",
                "pandas",
                "datasets"
            ],
            check=True,
        )


    _()
    return


@app.cell
def _():
    #sys.path.append(".")
    from shared.model_utils import load_model as lm
    model, tokenizer = lm(quantize=True)
    print('hi')
    return model, tokenizer


@app.cell
def _():
    CATEGORY = "ambiguity"   # or "ambiguity" / "contradictory_context"
    DATA_PATH = f"data/{CATEGORY}/prompts.jsonl"
    RESULTS_PATH = f"person_A_ambiguity/results/results.csv"  # adjust folder per person
    return CATEGORY, RESULTS_PATH


@app.cell
def _():
    import os
    import subprocess
    from pathlib import Path

    # 1. Check for tokens in environment
    HF_TOKEN = os.environ.get("HF_TOKEN")
    GH_TOKEN = os.environ.get("github_access")

    # 2. Build the correct, valid Git URL
    repo_user = "SherylMehta1"
    repo_name = "Confidence-Neurons-Across-Uncertainty-Types"

    if GH_TOKEN and GH_TOKEN.strip():
        clean_token = GH_TOKEN.strip()
        remote_url = f"https://{clean_token}@github.com/{repo_user}/{repo_name}.git"
        print("Secrets loaded: Authenticated Git URL configured.")
    else:
        remote_url = f"https://github.com/{repo_user}/{repo_name}.git"
        print("GITHUB_TOKEN_V2 not found in os.environ. Using standard public URL.")

    # 3. Fix ownership issues in cloud environments
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)

    # 4. Smart Directory & Git Management
    current_dir = Path.cwd()
    # If currently trapped in a nested double-folder, step out first
    if current_dir.name == repo_name and current_dir.parent.name == repo_name:
        os.chdir(current_dir.parent)
        current_dir = Path.cwd()

    # Case A: Already inside the correct repo directory
    if current_dir.name == repo_name and (current_dir / ".git").exists():
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=False)
        subprocess.run(["git", "pull"], check=False)
        print("Already inside repository. Updated remote & pulled latest changes.")

    # Case B: Repo exists as a subfolder
    elif (current_dir / repo_name / ".git").exists():
        os.chdir(current_dir / repo_name)
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=False)
        subprocess.run(["git", "pull"], check=False)
        print("Navigated into repository folder and pulled latest changes.")

    # Case C: Fresh Clone needed
    else:
        repo_path = current_dir / repo_name
        if repo_path.exists():
            import shutil
            shutil.rmtree(repo_path)
        subprocess.run(["git", "clone", remote_url], check=True)
        os.chdir(repo_path)
        print("Repository cloned successfully.")

    print(f"Current Working Directory: {Path.cwd()}")
    return os, shutil, subprocess


@app.cell
def _():
    '''import sys
    repo_path1 = os.getcwd()
    if repo_path not in sys.path:
        sys.path.append(repo_path)'''
    #import os

    import sys
    import importlib

    sys.path.insert(0, "/marimo/Confidence-Neurons-Across-Uncertainty-Types")

    # Remove the incorrectly loaded package
    sys.modules.pop("shared", None)

    # Import the local package again
    shared_module = importlib.import_module("shared")

    print(shared_module.__file__)
    return importlib, sys


@app.cell
def _(candidates_p2h):
    print("provenance:", candidates_p2h["provenance"])
    print("candidate type:", type(candidates_p2h["candidates"]))
    print("candidate count:", len(candidates_p2h["candidates"]))
    print("first candidate:", candidates_p2h["candidates"][0])
    return


@app.cell
def _(baseline_prompts):
    def _():
        import json as json_p2h
        import sys as sys_p2h
        sys_p2h.path.append(".")
        import pandas as pd_p2h
        from scipy.stats import wilcoxon as wilcoxon_p2h
        from shared.model_utils import load_model as load_model_p2h
        from shared.ablation import (
            compute_mean_activation as compute_mean_activation_p2h,
            run_ablation_experiment as run_ablation_experiment_p2h,
        )
        model_p2h, tokenizer_p2h = load_model_p2h(quantize=False)
        with open("candidate_neurons.json") as f_p2h:
            candidates_p2h = json_p2h.load(f_p2h)

        if isinstance(candidates_p2h, dict):
            candidates_p2h = candidates_p2h["candidates"]

    # Backward compatibility with the old list-of-lists format
        if candidates_p2h and isinstance(candidates_p2h[0], list):
            candidates_p2h = [
                {
                "layer": c[0],
                "neuron_idx": c[1],
                "neuron_id": f"L{c[0]}_N{c[1]}",
                }
                for c in candidates_p2h
            ]
        with open("data/ambiguity/prompts.jsonl") as f_prompts_p2h:
            all_prompts_p2h = [
                json_p2h.loads(line)
                for line in f_prompts_p2h
            ]
        working_prompts_p2h = [
            p
            for p in all_prompts_p2h
            if p["split"] == "working"
        ]
        held_out_prompts_p2h = [
            p
            for p in all_prompts_p2h
            if p["split"] == "held_out"
        ]
        assert len(held_out_prompts_p2h) == 36, (
            f"Expected 36 held-out prompts, got {len(held_out_prompts_p2h)} -- "
            f"held-out filter may still be broken"
        )
        comparison_rows_p2h = []
        for candidate_p2h in candidates_p2h:
            layer_idx_p2h = candidate_p2h["layer"]
            neuron_idx_p2h = candidate_p2h["neuron_idx"]
            mean_val_p2h = compute_mean_activation_p2h(
                model_p2h,
                tokenizer_p2h,
                baseline_prompts,
                layer_idx_p2h,
                neuron_idx_p2h,
            )
            working_rows_p2h = run_ablation_experiment_p2h(
                model_p2h,
                tokenizer_p2h,
                working_prompts_p2h,
                layer_idx_p2h,
                neuron_idx_p2h,
                mean_val_p2h,
                category="ambiguity",
                split="working",
            )
            heldout_rows_p2h = run_ablation_experiment_p2h(
                model_p2h,
                tokenizer_p2h,
                held_out_prompts_p2h,
                layer_idx_p2h,
                neuron_idx_p2h,
                mean_val_p2h,
                category="ambiguity",
                split="held_out",
            )
            w_df_p2h = pd_p2h.DataFrame(working_rows_p2h)
            h_df_p2h = pd_p2h.DataFrame(heldout_rows_p2h)
            w_stat_p2h, w_p_p2h = wilcoxon_p2h(
                w_df_p2h["orig_entropy"], w_df_p2h["ablated_entropy"]
            )
            h_stat_p2h, h_p_p2h = wilcoxon_p2h(
                h_df_p2h["orig_entropy"], h_df_p2h["ablated_entropy"]
            )
            comparison_rows_p2h.append({
                "neuron_id": candidate_p2h["neuron_id"],
                "working_mean_shift": w_df_p2h["entropy_shift"].mean(),
                "working_p": w_p_p2h,
                "working_n": len(w_df_p2h),
                "heldout_mean_shift": h_df_p2h["entropy_shift"].mean(),
                "heldout_p": h_p_p2h,
                "heldout_n": len(h_df_p2h),
                "replicates": (w_p_p2h < 0.01) and (h_p_p2h < 0.01),
            })
            print(
                f"{candidate_p2h['neuron_id']}: working p={w_p_p2h:.4f}, "
                f"held-out p={h_p_p2h:.4f}, "
                f"replicates={comparison_rows_p2h[-1]['replicates']}"
            )
        comparison_df_p2h = pd_p2h.DataFrame(comparison_rows_p2h)
        comparison_df_p2h.to_csv(
            "person_A_ambiguity/results/working_vs_heldout_newdata.csv",
            index=False,
        )
        return print(
            f"\n{comparison_df_p2h['replicates'].sum()} / {len(comparison_df_p2h)} "
            "candidates replicate on held-out"
        )


    _()
    return


@app.cell
def _():
    def _():
        import sys
        sys.path.append(".")  # run this script from the repo root
        import json
        import os
        import re

        from datasets import load_dataset
        from shared.schema_utils import load_tokenizer, build_records, save_records

        OUTPUT_PATH = "data/ambiguity/prompts.jsonl"
        REVIEW_LOG_PATH = "data/ambiguity/review_log.jsonl"

        # --- Stage 2: expanded factoid filter -----------------------------------
        EXCLUDE_OPEN_ENDED_PATTERNS = (
            r"^how (do|to|does|can|should|did|are|is)\b",
            r"^why\b",
            r"what are the (methods|steps|ways|effects|reasons|benefits|advantages|disadvantages|causes|consequences|implications)",
            r"^describe\b", r"^explain\b", r"^discuss\b",
            r"in what ways", r"to what extent",
        )

        ACCEPT_FACTOID_PATTERNS = (
            r"^(what|who|where|when|which|name|in which|in what year|how (many|much|old|long|far))\b",
        )

        def is_factoid_question(question: str) -> bool:
            q = question.strip().lower()
            for pattern in EXCLUDE_OPEN_ENDED_PATTERNS:
                if re.search(pattern, q):
                    return False
            for pattern in ACCEPT_FACTOID_PATTERNS:
                if re.search(pattern, q):
                    return True
            return False

        # --- Stage 2b: malformed / run-on / multi-question check ----------------
        def looks_malformed(question: str) -> bool:
            """Catch likely-garbled/multi-question run-ons that pattern-matching
            alone won't catch -- a cheap heuristic, not a full coherence check."""
            q = question.strip().lower()
            question_words = ["who", "what", "when", "where", "which", "how", "why"]
            word_count = q.split()
            n_question_words = sum(1 for w in word_count if w in question_words)
            if n_question_words >= 2:
                return True
            if len(word_count) > 20:  # ordinary factoid questions are rarely this long
                return True
            return False

        # --- Stage 3: genuine-ambiguity check ------------------------------------
        def normalize_answer(a: str) -> str:
            if not isinstance(a, str):
                return ""
            a = a.lower().strip()
            a = re.sub(r"[^\w\s]", "", a)
            a = re.sub(r"^(the|a|an)\s+", "", a)
            return a.strip()

        def extract_first_string(item) -> str:
            while isinstance(item, (list, tuple)) and len(item) > 0:
                item = item[0]
            return item if isinstance(item, str) else ""

        def has_genuinely_distinct_answers(answer_sets: list, min_distinct: int = 2) -> bool:
            normalized_reps = set()
            for group in answer_sets:
                if not group:
                    continue
                first_str = extract_first_string(group)
                norm = normalize_answer(first_str)
                if norm:
                    normalized_reps.add(norm)
            return len(normalized_reps) >= min_distinct

        def convert_to_cloze_prompt(question: str, suffix: str = " The answer is") -> str:
            q = question.strip()
            if not q.endswith("?"):
                q += "?"
            return q + suffix

        def load_ambigqa_records():
            ds = load_dataset("sewon/ambig_qa", "light", split="train")
            return ds

        def extract_question_and_answer_groups(record: dict):
            question = record.get("question")
            annotations = record.get("annotations")
            if question is None or annotations is None:
                return None

            answer_groups = []

            if isinstance(annotations, dict):
                ann_types = annotations.get("type", [])
                qa_pairs_list = annotations.get("qaPairs", [])

                for i, ann_type in enumerate(ann_types):
                    if ann_type == "multipleQAs" and i < len(qa_pairs_list):
                        qa_pair_dict = qa_pairs_list[i]
                        for ans_group in qa_pair_dict.get("answer", []):
                            if ans_group:
                                answer_groups.append(ans_group)

            return question.strip(), answer_groups

        def main():
            ds = load_ambigqa_records()
            print(f"Loaded {len(ds)} raw AmbigQA records.")

            review_log = []
            accepted_prompts = []

            for record in ds:
                parsed = extract_question_and_answer_groups(record)
                if parsed is None:
                    review_log.append({"question": None, "stage_failed": 0,
                                        "reason": "missing question/annotations field"})
                    continue
                question, answer_groups = parsed

                # Stage 1: AmbigQA's own multi-answer signal
                if len(answer_groups) < 2:
                    review_log.append({"question": question, "stage_failed": 1,
                                        "reason": "fewer than 2 answer groups (not ambiguous per AmbigQA)"})
                    continue

                # Stage 2: factoid structure
                if not is_factoid_question(question):
                    review_log.append({"question": question, "stage_failed": 2,
                                        "reason": "open-ended/explanatory question, no natural short answer"})
                    continue

                # Stage 2b: malformed / multi-question run-on
                if looks_malformed(question):
                    review_log.append({"question": question, "stage_failed": "2b",
                                        "reason": "malformed/run-on: multiple question-word clusters or too long"})
                    continue

                # Stage 3: genuinely distinct answers
                if not has_genuinely_distinct_answers(answer_groups, min_distinct=2):
                    review_log.append({"question": question, "stage_failed": 3,
                                        "reason": "answer groups are near-duplicates after normalization"})
                    continue

                review_log.append({"question": question, "stage_failed": None, "reason": "accepted"})
                accepted_prompts.append(convert_to_cloze_prompt(question))

            n_accepted = len(accepted_prompts)
            n_rejected = len(review_log) - n_accepted
            print(f"\nStage summary: {n_accepted} accepted, {n_rejected} rejected.")
            for stage in [0, 1, 2, "2b", 3]:
                n = sum(1 for r in review_log if r["stage_failed"] == stage)
                print(f"  rejected at stage {stage}: {n}")

            os.makedirs(os.path.dirname(REVIEW_LOG_PATH), exist_ok=True)
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

            with open(REVIEW_LOG_PATH, "w") as f:
                for entry in review_log:
                    f.write(json.dumps(entry) + "\n")
            print(f"Saved full review log ({len(review_log)} entries) to {REVIEW_LOG_PATH}")

            assert n_accepted >= 120, (
                f"Only {n_accepted} items survived all stages, need >= 120 to subsample."
            )

            tokenizer = load_tokenizer()
            records = build_records(
                raw_prompts=accepted_prompts,
                category="ambiguity",
                source_dataset="AmbigQA-light",
                prefix="amb",
                tokenizer=tokenizer,
                n_working=120,
                split_ratio=0.7,
            )
            save_records(records, OUTPUT_PATH)

        if __name__ == "__main__":
            return main()
    _()
    return


@app.cell
def _(baseline_prompts):
    def _():
        import json as json_p2
        import sys as sys_p2
        sys_p2.path.append(".")
        import pandas as pd_p2
        import torch as torch_p2
        from shared.model_utils import load_model as load_model_p2
        from shared.ablation import (
            compute_mean_activation as compute_mean_activation_p2,
            run_ablation_experiment as run_ablation_experiment_p2,
        )
        # ------------------------------------------------------------
        # 1. Load model at FULL bf16 precision -- no 4-bit quantization
        # ------------------------------------------------------------
        model_p2, tokenizer_p2 = load_model_p2(quantize=False)
        model_p2.eval()
        # ------------------------------------------------------------
        # 2. Identify candidates to re-run
        # ------------------------------------------------------------
        sig_df_p2 = pd_p2.read_csv(
            "person_A_ambiguity/results/significance_summary.csv"
        )
        priority_neurons_p2 = ["L26_N2788"]
        if "significant" in sig_df_p2.columns:
            near_misses_p2 = sig_df_p2[
                sig_df_p2["significant"] == True
            ]["neuron_id"].tolist()
            priority_neurons_p2 = list(
                set(priority_neurons_p2 + near_misses_p2)
            )
        print(f"Re-running at full precision: {priority_neurons_p2}")
        with open("candidate_neurons.json") as f_p2:
            all_candidates_p2 = json_p2.load(f_p2)
        # Normalize if still tuple/list format
        if isinstance(all_candidates_p2, list) and len(all_candidates_p2) > 0:
            # Check if the first element is a list/tuple or already a dict
            if isinstance(all_candidates_p2[0], (list, tuple)):
                all_candidates_p2 = [
                    {
                        "layer": c[0],
                        "neuron_idx": c[1],
                        "neuron_id": f"L{c[0]}_N{c[1]}",
                    }
                    for c in all_candidates_p2
                ]
            elif isinstance(all_candidates_p2[0], str):
                # Handle case where candidates are stored as strings like "L26_N2788"
                parsed_candidates_p2 = []
                for c in all_candidates_p2:
                    if isinstance(c, str) and c.startswith('L') and '_N' in c:
                        parts = c.replace('L', '').split('_N')
                        if len(parts) == 2:
                            try:
                                layer = int(parts[0])
                                neuron_idx = int(parts[1])
                                parsed_candidates_p2.append({
                                    "layer": layer,
                                    "neuron_idx": neuron_idx,
                                    "neuron_id": c,
                                })
                            except ValueError:
                                print(f"Warning: Could not parse candidate {c}")
                                continue
                all_candidates_p2 = parsed_candidates_p2
        candidates_to_rerun_p2 = [
            c
            for c in all_candidates_p2
            if c in priority_neurons_p2
        ]
        # ------------------------------------------------------------
        # 3. Load working AND held-out prompts
        # ------------------------------------------------------------
        with open("data/ambiguity/prompts.jsonl") as f_prompts_p2:
            all_prompts_p2 = [
                json_p2.loads(line)
                for line in f_prompts_p2
            ]
        working_prompts_p2 = [
            p
            for p in all_prompts_p2
            if p["split"] == "working"
        ]
        held_out_prompts_p2 = [
            p
            for p in all_prompts_p2
            if p["split"] == "held_out"
        ]
        print(
            f"Working: {len(working_prompts_p2)}, "
            f"Held-out: {len(held_out_prompts_p2)}"
        )
        assert len(held_out_prompts_p2) > 0, (
            "Held-out filter still broken -- fix Phase 1 first"
        )
        # ------------------------------------------------------------
        # 4. Run ablation at bf16 for both splits
        # ------------------------------------------------------------
        all_rows_p2 = []
        for candidate_p2 in candidates_to_rerun_p2:
            layer_idx_p2 = candidate_p2["layer"]
            neuron_idx_p2 = candidate_p2["neuron_idx"]
            mean_val_p2 = compute_mean_activation_p2(
                model_p2,
                tokenizer_p2,
                baseline_prompts,
                layer_idx_p2,
                neuron_idx_p2,
            )
            for split_name_p2, prompts_p2 in [
                ("working", working_prompts_p2),
                ("held_out", held_out_prompts_p2),
            ]:
                rows_p2 = run_ablation_experiment_p2(
                    model_p2,
                    tokenizer_p2,
                    prompts_p2,
                    layer_idx_p2,
                    neuron_idx_p2,
                    mean_val_p2,
                    category="ambiguity",
                    split=split_name_p2,
                )
                all_rows_p2.extend(rows_p2)
                print(
                    f"{candidate_p2['neuron_id']} "
                    f"[{split_name_p2}]: {len(rows_p2)} rows"
                )
        # ------------------------------------------------------------
        # 5. Save results -- new filename, new dataset
        # ------------------------------------------------------------
        df_bf16_p2 = pd_p2.DataFrame(all_rows_p2)
        df_bf16_p2.to_csv(
            "person_A_ambiguity/results/results_bf16_unquantized_newdata.csv",
            index=False,
        )
        return print(
            f"Saved {len(df_bf16_p2)} rows to "
            "person_A_ambiguity/results/results_bf16_unquantized_newdata.csv"
        )


    _()
    return


@app.cell
def _(baseline_prompts):
    def _():
        import json as json_p2h
        import sys as sys_p2h
        sys_p2h.path.append(".")
        import pandas as pd_p2h
        from scipy.stats import wilcoxon as wilcoxon_p2h
        from shared.model_utils import load_model as load_model_p2h
        from shared.ablation import (
            compute_mean_activation as compute_mean_activation_p2h,
            run_ablation_experiment as run_ablation_experiment_p2h,
        )
        model_p2h, tokenizer_p2h = load_model_p2h(quantize=False)
        with open("candidate_neurons.json") as f_p2h:
            candidates_p2h = json_p2h.load(f_p2h)
        if isinstance(candidates_p2h, list) and len(candidates_p2h) > 0:
            # Check if the first element is a list/tuple or already a dict
            if isinstance(candidates_p2h[0], (list, tuple)):
                candidates_p2h = [
                    {
                        "layer": c[0],
                        "neuron_idx": c[1],
                        "neuron_id": f"L{c[0]}_N{c[1]}",
                    }
                    for c in candidates_p2h
                ]
            elif isinstance(candidates_p2h[0], str):
                # Handle case where candidates are stored as strings like "L26_N2788"
                parsed_candidates_p2h = []
                for c in candidates_p2h:
                    if isinstance(c, str) and c.startswith('L') and '_N' in c:
                        parts = c.replace('L', '').split('_N')
                        if len(parts) == 2:
                            try:
                                layer = int(parts[0])
                                neuron_idx = int(parts[1])
                                parsed_candidates_p2h.append({
                                    "layer": layer,
                                    "neuron_idx": neuron_idx,
                                    "neuron_id": c,
                                })
                            except ValueError:
                                print(f"Warning: Could not parse candidate {c}")
                                continue
                candidates_p2h = parsed_candidates_p2h
        with open("data/ambiguity/prompts.jsonl") as f_prompts_p2h:
            all_prompts_p2h = [
                json_p2h.loads(line)
                for line in f_prompts_p2h
            ]
        working_prompts_p2h = [
            p
            for p in all_prompts_p2h
            if p["split"] == "working"
        ]
        held_out_prompts_p2h = [
            p
            for p in all_prompts_p2h
            if p["split"] == "held_out"
        ]
        assert len(held_out_prompts_p2h) == 36, (
            f"Expected 36 held-out prompts, got {len(held_out_prompts_p2h)} -- "
            f"held-out filter may still be broken"
        )
        comparison_rows_p2h = []
        for candidate_p2h in candidates_p2h:
            layer_idx_p2h = candidate_p2h["layer"]
            neuron_idx_p2h = candidate_p2h["neuron_idx"]
            mean_val_p2h = compute_mean_activation_p2h(
                model_p2h,
                tokenizer_p2h,
                baseline_prompts,
                layer_idx_p2h,
                neuron_idx_p2h,
            )
            working_rows_p2h = run_ablation_experiment_p2h(
                model_p2h,
                tokenizer_p2h,
                working_prompts_p2h,
                layer_idx_p2h,
                neuron_idx_p2h,
                mean_val_p2h,
                category="ambiguity",
                split="working",
            )
            heldout_rows_p2h = run_ablation_experiment_p2h(
                model_p2h,
                tokenizer_p2h,
                held_out_prompts_p2h,
                layer_idx_p2h,
                neuron_idx_p2h,
                mean_val_p2h,
                category="ambiguity",
                split="held_out",
            )
            w_df_p2h = pd_p2h.DataFrame(working_rows_p2h)
            h_df_p2h = pd_p2h.DataFrame(heldout_rows_p2h)
            w_stat_p2h, w_p_p2h = wilcoxon_p2h(
                w_df_p2h["orig_entropy"], w_df_p2h["ablated_entropy"]
            )
            h_stat_p2h, h_p_p2h = wilcoxon_p2h(
                h_df_p2h["orig_entropy"], h_df_p2h["ablated_entropy"]
            )
            comparison_rows_p2h.append({
                "neuron_id": candidate_p2h["neuron_id"],
                "working_mean_shift": w_df_p2h["entropy_shift"].mean(),
                "working_p": w_p_p2h,
                "working_n": len(w_df_p2h),
                "heldout_mean_shift": h_df_p2h["entropy_shift"].mean(),
                "heldout_p": h_p_p2h,
                "heldout_n": len(h_df_p2h),
                "replicates": (w_p_p2h < 0.01) and (h_p_p2h < 0.01),
            })
            print(
                f"{candidate_p2h['neuron_id']}: working p={w_p_p2h:.4f}, "
                f"held-out p={h_p_p2h:.4f}, "
                f"replicates={comparison_rows_p2h[-1]['replicates']}"
            )
        comparison_df_p2h = pd_p2h.DataFrame(comparison_rows_p2h)
        comparison_df_p2h.to_csv(
            "person_A_ambiguity/results/working_vs_heldout_newdata.csv",
            index=False,
        )
        return print(
            f"\n{comparison_df_p2h['replicates'].sum()} / {len(comparison_df_p2h)} "
            "candidates replicate on held-out"
        )


    _()
    return


@app.cell
def _(baseline_prompts):
    import json as json_fnc2
    import sys as sys_fnc2

    sys_fnc2.path.append(".")

    import torch as torch_fnc2
    import pandas as pd_fnc2

    from shared.model_utils import (
        load_model as load_model_fnc2,
        compute_entropy as compute_entropy_fnc2,
    )

    from shared.ablation import (
        compute_mean_activation as compute_mean_activation_fnc2
    )


    # ============================================================
    # 1. Load Model and Tokenizer
    # ============================================================

    model_fnc2, tokenizer_fnc2 = load_model_fnc2(quantize=False)


    # ============================================================
    # 2. Frozen Norm Ablation Function
    # ============================================================

    def frozen_norm_ablate_fnc2(
        model_fnc2,
        tokenizer_fnc2,
        prompt_text_fnc2,
        layer_idx_fnc2,
        neuron_idx_fnc2,
        mean_val_fnc2
    ):

        inputs_fnc2 = tokenizer_fnc2(prompt_text_fnc2, return_tensors="pt", add_special_tokens=False).to(model_fnc2.device)

        captured_fnc2 = {}

        def capture_hook_fnc2(module, input, output):
            x_fnc2 = input[0].float()
            variance_fnc2 = x_fnc2.pow(2).mean(-1, keepdim=True)
            rms_fnc2 = torch_fnc2.rsqrt(variance_fnc2 + module.variance_epsilon)
            captured_fnc2["inv_rms"] = rms_fnc2.detach()

        norm_handle_fnc2 = model_fnc2.model.norm.register_forward_hook(capture_hook_fnc2)

        with torch_fnc2.no_grad():
            _ = model_fnc2(**inputs_fnc2, use_cache=False)

        norm_handle_fnc2.remove()

        down_proj_fnc2 = model_fnc2.model.layers[layer_idx_fnc2].mlp.down_proj

        def ablate_hook_fnc2(module, args):
            modified_fnc2 = args[0].clone()
            modified_fnc2[:, :, neuron_idx_fnc2] = mean_val_fnc2
            return (modified_fnc2,) + args[1:]

        def frozen_norm_hook_fnc2(module, input, output):
            x_fnc2 = input[0].float()
            inv_rms_fnc2 = captured_fnc2["inv_rms"]
            normed_fnc2 = (x_fnc2 * inv_rms_fnc2).to(module.weight.dtype) * module.weight
            return normed_fnc2

        ablate_handle_fnc2 = down_proj_fnc2.register_forward_pre_hook(ablate_hook_fnc2)
        norm_freeze_handle_fnc2 = model_fnc2.model.norm.register_forward_hook(frozen_norm_hook_fnc2)

        try:
            with torch_fnc2.no_grad():
                outputs_fnc2 = model_fnc2(**inputs_fnc2, use_cache=False)
        finally:
            ablate_handle_fnc2.remove()
            norm_freeze_handle_fnc2.remove()

        logits_fnc2 = outputs_fnc2.logits[0, -1, :].float()
        return torch_fnc2.nn.functional.softmax(logits_fnc2, dim=-1)


    # ============================================================
    # 3. Load AMBIGUITY Prompts (new dataset, same path)
    # ============================================================

    with open("data/ambiguity/prompts.jsonl") as f_fnc2:
        all_prompts_fnc2 = [json_fnc2.loads(line) for line in f_fnc2]

    working_prompts_fnc2 = [p_fnc2 for p_fnc2 in all_prompts_fnc2 if p_fnc2["split"] == "working"]


    # ============================================================
    # 4. Define Target Layer/Neuron -- L31_N2477
    # ============================================================

    layer_idx_fnc2 = 31
    neuron_idx_fnc2 = 2477


    # ============================================================
    # 5. Compute Mean Baseline Activation
    # ============================================================

    mean_val_fnc2 = compute_mean_activation_fnc2(
        model_fnc2, tokenizer_fnc2, baseline_prompts, layer_idx_fnc2, neuron_idx_fnc2
    )


    # ============================================================
    # 6. Run Precision-Controlled Experiment Loop
    # ============================================================

    results_fnc2 = []

    for p_fnc2 in working_prompts_fnc2:
        prompt_text_fnc2 = p_fnc2["chat_formatted_prompt"]

        # Baseline forward pass -- add_special_tokens=False to MATCH
        # the frozen-norm pass's tokenization exactly
        inputs_fnc2 = tokenizer_fnc2(prompt_text_fnc2, return_tensors="pt", add_special_tokens=False).to(model_fnc2.device)

        with torch_fnc2.no_grad():
            orig_out_fnc2 = model_fnc2(**inputs_fnc2, use_cache=False)

        orig_logits_fnc2 = orig_out_fnc2.logits[0, -1, :].float()
        orig_probs_fnc2 = torch_fnc2.nn.functional.softmax(orig_logits_fnc2, dim=-1)
        orig_entropy_fnc2 = compute_entropy_fnc2(orig_probs_fnc2)

        frozen_probs_fnc2 = frozen_norm_ablate_fnc2(
            model_fnc2, tokenizer_fnc2, prompt_text_fnc2, layer_idx_fnc2, neuron_idx_fnc2, mean_val_fnc2
        )
        frozen_entropy_fnc2 = compute_entropy_fnc2(frozen_probs_fnc2)

        results_fnc2.append({
            "prompt_id": p_fnc2["prompt_id"],
            "orig_entropy": orig_entropy_fnc2,
            "frozen_norm_ablated_entropy": frozen_entropy_fnc2,
            "shift_under_frozen_norm": frozen_entropy_fnc2 - orig_entropy_fnc2,
        })


    # ============================================================
    # 7. Output to CSV -- new filename, new dataset
    # ============================================================

    df_fnc2 = pd_fnc2.DataFrame(results_fnc2)
    df_fnc2.to_csv("person_A_ambiguity/results/frozen_norm_L31_N2477_newdata.csv", index=False)


    # ============================================================
    # 8. Output Summary
    # ============================================================

    print(f"Mean shift under frozen norm: {df_fnc2['shift_under_frozen_norm'].mean():.6f}")
    print("Saved to: person_A_ambiguity/results/frozen_norm_L31_N2477_newdata.csv")
    return


@app.cell
def _(pd):
    df_ambig_frozen = pd.read_csv("person_A_ambiguity/results/frozen_norm_L31_N2477_newdata.csv")
    print(df_ambig_frozen["shift_under_frozen_norm"].describe())
    return


@app.cell
def _():
    return


@app.cell
def _(subprocess):
    #import subprocess

    # ------------------------------------------------------------
    # 1. Pull latest first
    # ------------------------------------------------------------
    pull_result = subprocess.run(["git", "pull"], capture_output=True, text=True)
    print(pull_result.stdout, pull_result.stderr)
    return


@app.cell
def _(shutil):


    shutil.copy(
        "person_A_ambiguity/results/results_bf16_unquantized_newdata.csv",
        "person_A_ambiguity/results/results_bf16_unquantized.csv",
    )
    shutil.copy(
        "person_A_ambiguity/results/working_vs_heldout_newdata.csv",
        "person_A_ambiguity/results/working_vs_heldout.csv",
    )
    shutil.copy(
        "person_A_ambiguity/results/frozen_norm_L31_N2477_newdata.csv",
        "person_A_ambiguity/results/frozen_norm_L31_N2477.csv",
    )
    print("Copied _newdata files to standard filenames")
    return


@app.cell
def _(subprocess):
    status_result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    print(status_result.stdout)
    return


@app.cell
def _(subprocess):


    add_result = subprocess.run(
        ["git", "add",
         "person_A_ambiguity/results/results_bf16_unquantized.csv",
         "person_A_ambiguity/results/working_vs_heldout.csv",
         "person_A_ambiguity/results/frozen_norm_L31_N2477.csv"],
        capture_output=True, text=True
    )
    print(add_result.stdout, add_result.stderr)

    staged_check = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    print(staged_check.stdout)
    return


@app.cell
def _(subprocess):


    commit_result = subprocess.run(
        ["git", "commit", "-m",
         "Phase 2 ambiguity results on corrected dataset: bf16 rerun, held-out eval, frozen-norm L31_N2477"],
        capture_output=True, text=True
    )
    print(commit_result.stdout, commit_result.stderr)
    return


@app.cell
def _(subprocess):
    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    print(push_result.stdout, push_result.stderr)
    return


@app.cell
def _(subprocess):
    def _():
        status_result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
        return print(status_result.stdout)


    _()
    return


@app.cell
def _(subprocess):
    add_result1 = subprocess.run(
        ["git", "add", "data/ambiguity/prompts.jsonl", "data/ambiguity/review_log.jsonl"],
        capture_output=True, text=True
    )
    print(add_result1.stdout, add_result1.stderr)
    return


@app.cell
def _(subprocess):
    commit_result1 = subprocess.run(
        ["git", "commit", "-m",
         "Phase 2 remediation: rebuild ambiguity prompts.jsonl with fixed AmbigQA extraction + malformed-question filter"],
        capture_output=True, text=True
    )
    print(commit_result1.stdout, commit_result1.stderr)
    return


@app.cell
def _(subprocess):

    push_result1 = subprocess.run(["git", "push"], capture_output=True, text=True)
    print(push_result1.stdout, push_result1.stderr)
    return


@app.cell
def _(subprocess):
    result11 = subprocess.run(
        ["grep", "-r", "preprocess_ambiguity", ".", "--include=*.py"],
        capture_output=True, text=True
    )
    print(result11.stdout)
    return


@app.cell
def _(os):

    os.makedirs("person_A_ambiguity/preprocessing_ambiguity_2", exist_ok=True)
    return


@app.cell
def _(os):


    code_content = """
    import sys
    sys.path.append(".")
    import json
    import os
    import re

    from datasets import load_dataset
    from shared.schema_utils import load_tokenizer, build_records, save_records

    OUTPUT_PATH = "data/ambiguity/prompts.jsonl"
    REVIEW_LOG_PATH = "data/ambiguity/review_log.jsonl"

    # --- Stage 2: factoid filter ---------------------------------------------
    EXCLUDE_OPEN_ENDED_PATTERNS = (
        r"^how (do|to|does|can|should|did|are|is)\\b",
        r"^why\\b",
        r"what are the (methods|steps|ways|effects|reasons|benefits|advantages|disadvantages|causes|consequences|implications)",
        r"^describe\\b", r"^explain\\b", r"^discuss\\b",
        r"in what ways", r"to what extent",
    )

    ACCEPT_FACTOID_PATTERNS = (
        r"^(what|who|where|when|which|name|in which|in what year|how (many|much|old|long|far))\\b",
    )

    def is_factoid_question(question: str) -> bool:
        q = question.strip().lower()
        for pattern in EXCLUDE_OPEN_ENDED_PATTERNS:
            if re.search(pattern, q):
                return False
        for pattern in ACCEPT_FACTOID_PATTERNS:
            if re.search(pattern, q):
                return True
        return False

    def looks_malformed(question: str) -> bool:
        q = question.strip().lower()
        question_words = ["who", "what", "when", "where", "which", "how", "why"]
        word_count = q.split()
        n_question_words = sum(1 for w in word_count if w in question_words)
        if n_question_words >= 2:
            return True
        if len(word_count) > 20:
            return True
        return False

    def normalize_answer(a: str) -> str:
        if not isinstance(a, str):
            return ""
        a = a.lower().strip()
        a = re.sub(r"[^\\w\\s]", "", a)
        a = re.sub(r"^(the|a|an)\\s+", "", a)
        return a.strip()

    def extract_first_string(item) -> str:
        while isinstance(item, (list, tuple)) and len(item) > 0:
            item = item[0]
        return item if isinstance(item, str) else ""

    def has_genuinely_distinct_answers(answer_sets: list, min_distinct: int = 2) -> bool:
        normalized_reps = set()
        for group in answer_sets:
            if not group:
                continue
            first_str = extract_first_string(group)
            norm = normalize_answer(first_str)
            if norm:
                normalized_reps.add(norm)
        return len(normalized_reps) >= min_distinct

    def convert_to_cloze_prompt(question: str, suffix: str = " The answer is") -> str:
        q = question.strip()
        if not q.endswith("?"):
            q += "?"
        return q + suffix

    def load_ambigqa_records():
        ds = load_dataset("sewon/ambig_qa", "light", split="train")
        return ds

    def extract_question_and_answer_groups(record: dict):
        question = record.get("question")
        annotations = record.get("annotations")
        if question is None or annotations is None:
            return None

        answer_groups = []
        if isinstance(annotations, dict):
            ann_types = annotations.get("type", [])
            qa_pairs_list = annotations.get("qaPairs", [])

            for i, ann_type in enumerate(ann_types):
                if ann_type == "multipleQAs" and i < len(qa_pairs_list):
                    qa_pair_dict = qa_pairs_list[i]
                    for ans_group in qa_pair_dict.get("answer", []):
                        if ans_group:
                            answer_groups.append(ans_group)

        return question.strip(), answer_groups

    def main():
        ds = load_ambigqa_records()
        print(f"Loaded {len(ds)} raw AmbigQA records.")

        review_log = []
        accepted_prompts = []

        for record in ds:
            parsed = extract_question_and_answer_groups(record)
            if parsed is None:
                review_log.append({"question": None, "stage_failed": 0, "reason": "missing question/annotations field"})
                continue
            question, answer_groups = parsed

            if len(answer_groups) < 2:
                review_log.append({"question": question, "stage_failed": 1, "reason": "fewer than 2 answer groups (not ambiguous per AmbigQA)"})
                continue

            if not is_factoid_question(question):
                review_log.append({"question": question, "stage_failed": 2, "reason": "open-ended/explanatory question, no natural short answer"})
                continue

            if looks_malformed(question):
                review_log.append({"question": question, "stage_failed": "2b", "reason": "malformed/run-on: multiple question-word clusters or too long"})
                continue

            if not has_genuinely_distinct_answers(answer_groups, min_distinct=2):
                review_log.append({"question": question, "stage_failed": 3, "reason": "answer groups are near-duplicates after normalization"})
                continue

            review_log.append({"question": question, "stage_failed": None, "reason": "accepted"})
            accepted_prompts.append(convert_to_cloze_prompt(question))

        n_accepted = len(accepted_prompts)
        n_rejected = len(review_log) - n_accepted
        print(f"\\nStage summary: {n_accepted} accepted, {n_rejected} rejected.")
        for stage in [0, 1, 2, "2b", 3]:
            n = sum(1 for r in review_log if r["stage_failed"] == stage)
            print(f"  rejected at stage {stage}: {n}")

        os.makedirs(os.path.dirname(REVIEW_LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        with open(REVIEW_LOG_PATH, "w") as f:
            for entry in review_log:
                f.write(json.dumps(entry) + "\\n")
        print(f"Saved full review log ({len(review_log)} entries) to {REVIEW_LOG_PATH}")

        assert n_accepted >= 120, f"Only {n_accepted} items survived all stages, need >= 120 to subsample."

        tokenizer = load_tokenizer()
        records = build_records(
            raw_prompts=accepted_prompts,
            category="ambiguity",
            source_dataset="AmbigQA-light",
            prefix="amb",
            tokenizer=tokenizer,
            n_working=120,
            split_ratio=0.7,
        )
        save_records(records, OUTPUT_PATH)

    if __name__ == "__main__":
        main()
    """

    target_path = "person_A_ambiguity/preprocessing_ambiguity_2/preprocess_ambiguity_v2.py"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(code_content.strip())

    print(f"Successfully saved to {target_path}")
    return


@app.cell
def _(subprocess):


    # 1. Stage the file
    add_result2 = subprocess.run(
        ["git", "add", "person_A_ambiguity/preprocessing_ambiguity_2/preprocess_ambiguity_v2.py"],
        capture_output=True, text=True
    )
    print("Git Add Output:", add_result2.stdout, add_result2.stderr)

    # 2. Commit the changes
    commit_result2 = subprocess.run(
        ["git", "commit", "-m", "Add Phase 2 remediation preprocessing: fixed AmbigQA extraction bug + malformed filter"],
        capture_output=True, text=True
    )
    print("Git Commit Output:", commit_result2.stdout, commit_result2.stderr)

    # 3. Push to your remote repository
    push_result2 = subprocess.run(["git", "push"], capture_output=True, text=True)
    print("Git Push Output:", push_result2.stdout, push_result2.stderr)
    return


@app.cell
def _(subprocess):
    pull_result3 = subprocess.run(["git", "pull", "--rebase"], capture_output=True, text=True)
    print("Git Pull Output:\n", pull_result3.stdout, pull_result3.stderr)

    # 2. Push your changes again
    push_result4 = subprocess.run(["git", "push"], capture_output=True, text=True)
    print("Git Push Output:\n", push_result4.stdout, push_result4.stderr)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(subprocess):
    #import subprocess

    print(subprocess.run(["git", "--version"], capture_output=True, text=True).stdout)
    return


@app.cell
def _(subprocess):
    #import subprocess

    print(
        subprocess.run(
            ["git", "status"],
            cwd="/marimo/Confidence-Neurons-Across-Uncertainty-Types",
            capture_output=True,
            text=True,
        ).stdout
    )
    return


@app.cell
def _(os):
    #import os

    repo = "/marimo/Confidence-Neurons-Across-Uncertainty-Types"

    print(os.path.exists(f"{repo}/B_marimo_nb.py"))
    print(os.path.exists(f"{repo}/C_marimo_nb.py"))
    return


@app.cell
def _(subprocess):
    #import subprocess

    print(
        subprocess.run(
            ["git", "status", "--untracked-files=all"],
            cwd="/marimo/Confidence-Neurons-Across-Uncertainty-Types",
            capture_output=True,
            text=True,
        ).stdout
    )
    return


@app.cell
def _(os):
    #import os

    print(os.listdir("/marimo/Confidence-Neurons-Across-Uncertainty-Types")[:30])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
