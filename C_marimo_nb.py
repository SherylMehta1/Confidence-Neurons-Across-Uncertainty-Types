import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import subprocess
    from pathlib import Path

    # 1. Check for tokens in environment
    HF_TOKEN = os.environ.get("HF_TOKEN")
    GH_TOKEN = os.environ.get("GITHUB_TOKEN_V2")

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
    return HF_TOKEN, os, subprocess


@app.cell
def _(os):


    # Move up one directory level
    if os.getcwd().endswith("Confidence-Neurons-Across-Uncertainty-Types/Confidence-Neurons-Across-Uncertainty-Types"):
        os.chdir("..")

    print(f"✅ Corrected Working Directory: {os.getcwd()}")
    return


@app.cell
def _(HF_TOKEN):
    HF_TOKEN is not None
    return


@app.cell
def _(subprocess):
    # Cell 2: install packages
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "transformers", "accelerate", "bitsandbytes", "huggingface_hub"],
        check=True,
    )
    return (sys,)


@app.cell
def _(os):
    # Cell 3: HF login
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])
    return


@app.cell
def _(sys):
    # Cell 4: load model

    sys.path.append(".")
    from shared.model_utils import load_model
    model, tokenizer = load_model(quantize=True)
    return model, tokenizer


@app.cell
def _(subprocess):
    remote_check = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    print(remote_check.stdout, remote_check.stderr)
    return


@app.cell
def _(os, subprocess):


    # 1. Grab token from environment
    gh_token = os.environ["GITHUB_TOKEN_V2"]

    # 2. Re-attach token to origin URL
    remote_set_result = subprocess.run(
        ["git", "remote", "set-url", "origin", 
         f"https://{gh_token}@github.com/SherylMehta1/Confidence-Neurons-Across-Uncertainty-Types.git"],
        capture_output=True, text=True
    )

    print(remote_set_result.stdout, remote_set_result.stderr)
    return


@app.cell
def _(subprocess):
    remote_check2 = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    print(remote_check2.stdout, remote_check2.stderr)
    return


@app.cell
def _():
    CATEGORY = "contradictory_context"
    DATA_PATH = f"data/{CATEGORY}/prompts.jsonl"
    RESULTS_PATH = f"person_C_contradictory_context/results/results.csv"
    return CATEGORY, DATA_PATH, RESULTS_PATH


@app.cell
def _():
    from shared.detection import load_candidate_neurons
    candidates = load_candidate_neurons("candidate_neurons.json")
    print(f"{len(candidates)} candidates loaded")

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
    return baseline_prompts, candidates


@app.cell
def _(DATA_PATH):
    import json
    with open(DATA_PATH) as f:
        all_prompts = [json.loads(line) for line in f]

    working_prompts = [p for p in all_prompts if p["split"] == "working"]
    held_out_prompts = [p for p in all_prompts if p["split"] == "held_out"]

    print(f"working: {len(working_prompts)}, held_out: {len(held_out_prompts)}")
    assert len(held_out_prompts) > 0, "No held_out prompts found -- check the split field in your prompts.jsonl"
    return held_out_prompts, json, working_prompts


@app.cell
def _(
    CATEGORY,
    baseline_prompts,
    candidates,
    held_out_prompts,
    model,
    tokenizer,
    working_prompts,
):
    import pandas as pd
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
def _(candidates, json, model, tokenizer):
    from shared.logit_lens import direct_effect_score, top_direct_effect_tokens


    mechanism_results = []
    for c in candidates:
        layer_idx, neuron_idx = c["layer"], c["neuron_idx"]
        score = direct_effect_score(model, layer_idx, neuron_idx)
        top_tokens = top_direct_effect_tokens(model, tokenizer, layer_idx, neuron_idx, k=5)
        mechanism_results.append({
            "neuron_id": c["neuron_id"], "layer": layer_idx, "neuron_idx": neuron_idx,
            "direct_effect_score": score, "top_tokens": top_tokens,
        })
        print(f"{c['neuron_id']}: direct_effect_score={score:.4f}, top_tokens={top_tokens}")

    with open("mechanism_check_shared.json", "w") as mechanism_check_file:
        json.dump(mechanism_results, mechanism_check_file, indent=2)
    print("Saved mechanism_check_shared.json (RMSNorm-corrected)")
    return


@app.cell
def _(subprocess):

    result = subprocess.run(["git", "status"], capture_output=True, text=True)
    print(result.stdout)
    return


@app.cell
def _(subprocess):
    add_result = subprocess.run(
        ["git", "add",
         "person_C_contradictory_context/results/results.csv",
         "mechanism_check_shared.json"],
        capture_output=True, text=True
    )
    print(add_result.stdout, add_result.stderr)
    return


@app.cell
def _(subprocess):
    commit_result = subprocess.run(
        ["git", "commit", "-m",
         "Update Phase 4 results and RMSNorm-corrected mechanism check for contradictory_context (post audit, pre Phase 2 re-measurement)"],
        capture_output=True, text=True
    )
    print(commit_result.stdout, commit_result.stderr)
    return


@app.cell
def _(subprocess):
    pull_result = subprocess.run(["git", "pull", "--rebase"], capture_output=True, text=True)
    print(pull_result.stdout, pull_result.stderr)
    return


@app.cell
def _(subprocess):
    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    print(push_result.stdout, push_result.stderr)
    return


@app.cell
def _(subprocess):
    ls_result = subprocess.run(["find", ".", "-name", "mechanism_check_shared.json"], capture_output=True, text=True)
    print(ls_result.stdout, ls_result.stderr)
    return


@app.cell
def _(subprocess):
    final_remote_check = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    print(final_remote_check.stdout)
    return


@app.cell
def _(subprocess):
    test_pull = subprocess.run(["git", "pull"], capture_output=True, text=True)
    print(test_pull.stdout, test_pull.stderr)
    return


@app.cell
def _(subprocess):
    verify_pull = subprocess.run(["git", "pull"], capture_output=True, text=True)
    print(verify_pull.stdout, verify_pull.stderr)
    return


@app.cell
def _(baseline_prompts):
    # Phase 2: unquantized bf16 rerun -- contradictory_context
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
        "person_C_contradictory_context/results/significance_summary.csv"
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
    if isinstance(all_candidates_p2[0], list):
        all_candidates_p2 = [
            {
                "layer": c[0],
                "neuron_idx": c[1],
                "neuron_id": f"L{c[0]}_N{c[1]}",
            }
            for c in all_candidates_p2
        ]
    candidates_to_rerun_p2 = [
        c
        for c in all_candidates_p2
        if c["neuron_id"] in priority_neurons_p2
    ]
    # ------------------------------------------------------------
    # 3. Load working AND held-out prompts
    # ------------------------------------------------------------
    with open("data/contradictory_context/prompts.jsonl") as f_prompts_p2:
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
                category="contradictory_context",
                split=split_name_p2,
            )
            all_rows_p2.extend(rows_p2)
            print(
                f"{candidate_p2['neuron_id']} "
                f"[{split_name_p2}]: {len(rows_p2)} rows"
            )
    # ------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------
    df_bf16_p2 = pd_p2.DataFrame(all_rows_p2)
    df_bf16_p2.to_csv(
        "person_C_contradictory_context/results/results_bf16_unquantized.csv",
        index=False,
    )
    print(
        f"Saved {len(df_bf16_p2)} rows to "
        "person_C_contradictory_context/results/results_bf16_unquantized.csv"
    )
    return


@app.cell
def _(baseline_prompts):
    # Phase 2: held-out evaluation -- contradictory_context
    import json as json_p2hc
    import sys as sys_p2hc
    sys_p2hc.path.append(".")
    import pandas as pd_p2hc
    from scipy.stats import wilcoxon as wilcoxon_p2hc

    from shared.model_utils import load_model as load_model_p2hc
    from shared.ablation import (
        compute_mean_activation as compute_mean_activation_p2hc,
        run_ablation_experiment as run_ablation_experiment_p2hc,
    )

    model_p2hc, tokenizer_p2hc = load_model_p2hc(quantize=False)

    with open("candidate_neurons.json") as f_p2hc:
        candidates_p2hc = json_p2hc.load(f_p2hc)
    if isinstance(candidates_p2hc[0], list):
        candidates_p2hc = [
            {
                "layer": c[0],
                "neuron_idx": c[1],
                "neuron_id": f"L{c[0]}_N{c[1]}",
            }
            for c in candidates_p2hc
        ]

    with open("data/contradictory_context/prompts.jsonl") as f_prompts_p2hc:
        all_prompts_p2hc = [
            json_p2hc.loads(line)
            for line in f_prompts_p2hc
        ]

    working_prompts_p2hc = [
        p
        for p in all_prompts_p2hc
        if p["split"] == "working"
    ]
    held_out_prompts_p2hc = [
        p
        for p in all_prompts_p2hc
        if p["split"] == "held_out"
    ]

    assert len(held_out_prompts_p2hc) == 36, (
        f"Expected 36 held-out prompts, got {len(held_out_prompts_p2hc)} -- "
        f"held-out filter may still be broken"
    )

    comparison_rows_p2hc = []
    for candidate_p2hc in candidates_p2hc:
        layer_idx_p2hc = candidate_p2hc["layer"]
        neuron_idx_p2hc = candidate_p2hc["neuron_idx"]
        mean_val_p2hc = compute_mean_activation_p2hc(
            model_p2hc,
            tokenizer_p2hc,
            baseline_prompts,
            layer_idx_p2hc,
            neuron_idx_p2hc,
        )

        working_rows_p2hc = run_ablation_experiment_p2hc(
            model_p2hc,
            tokenizer_p2hc,
            working_prompts_p2hc,
            layer_idx_p2hc,
            neuron_idx_p2hc,
            mean_val_p2hc,
            category="contradictory_context",
            split="working",
        )
        heldout_rows_p2hc = run_ablation_experiment_p2hc(
            model_p2hc,
            tokenizer_p2hc,
            held_out_prompts_p2hc,
            layer_idx_p2hc,
            neuron_idx_p2hc,
            mean_val_p2hc,
            category="contradictory_context",
            split="held_out",
        )

        w_df_p2hc = pd_p2hc.DataFrame(working_rows_p2hc)
        h_df_p2hc = pd_p2hc.DataFrame(heldout_rows_p2hc)

        w_stat_p2hc, w_p_p2hc = wilcoxon_p2hc(
            w_df_p2hc["orig_entropy"], w_df_p2hc["ablated_entropy"]
        )
        h_stat_p2hc, h_p_p2hc = wilcoxon_p2hc(
            h_df_p2hc["orig_entropy"], h_df_p2hc["ablated_entropy"]
        )

        comparison_rows_p2hc.append({
            "neuron_id": candidate_p2hc["neuron_id"],
            "working_mean_shift": w_df_p2hc["entropy_shift"].mean(),
            "working_p": w_p_p2hc,
            "working_n": len(w_df_p2hc),
            "heldout_mean_shift": h_df_p2hc["entropy_shift"].mean(),
            "heldout_p": h_p_p2hc,
            "heldout_n": len(h_df_p2hc),
            "replicates": (w_p_p2hc < 0.01) and (h_p_p2hc < 0.01),
        })
        print(
            f"{candidate_p2hc['neuron_id']}: working p={w_p_p2hc:.4f}, "
            f"held-out p={h_p_p2hc:.4f}, "
            f"replicates={comparison_rows_p2hc[-1]['replicates']}"
        )

    comparison_df_p2hc = pd_p2hc.DataFrame(comparison_rows_p2hc)
    comparison_df_p2hc.to_csv(
        "person_C_contradictory_context/results/working_vs_heldout.csv",
        index=False,
    )
    print(
        f"\n{comparison_df_p2hc['replicates'].sum()} / {len(comparison_df_p2hc)} "
        "candidates replicate on held-out"
    )
    return


@app.cell
def _(baseline_prompts):
    # Phase 2: frozen-norm ablation -- contradictory_context, testing L31_N2477 - fixed
    import json as json_fnc
    import sys as sys_fnc
    sys_fnc.path.append(".")

    import torch as torch_fnc
    import pandas as pd_fnc
    from shared.model_utils import (
        load_model as load_model_fnc,
        compute_entropy as compute_entropy_fnc,
    )
    from shared.ablation import compute_mean_activation as compute_mean_activation_fnc

    # 1. Load Model and Tokenizer
    model_fnc, tokenizer_fnc = load_model_fnc(quantize=False)

    # 2. Fixed Frozen-Norm Ablation Function
    def frozen_norm_ablate_fnc(model_fnc, tokenizer_fnc, prompt_text_fnc, layer_idx_fnc, neuron_idx_fnc, mean_val_fnc):
        inputs_fnc = tokenizer_fnc(prompt_text_fnc, return_tensors="pt", add_special_tokens=False).to(model_fnc.device)

        # Capture RMSNorm denominator in float32 matching HF LlamaRMSNorm exactly
        captured_fnc = {}
        def capture_hook_fnc(module, input, output):
            x_fnc = input[0].float()
            variance_fnc = x_fnc.pow(2).mean(-1, keepdim=True)
            rms_fnc = torch_fnc.rsqrt(variance_fnc + module.variance_epsilon)
            captured_fnc["inv_rms"] = rms_fnc.detach()

        norm_handle_fnc = model_fnc.model.norm.register_forward_hook(capture_hook_fnc)
        with torch_fnc.no_grad():
            _ = model_fnc(**inputs_fnc, use_cache=False)
        norm_handle_fnc.remove()

        # Setup hooks for neuron ablation & frozen normalization
        down_proj_fnc = model_fnc.model.layers[layer_idx_fnc].mlp.down_proj

        def ablate_hook_fnc(module, args):
            modified_fnc = args[0].clone()
            modified_fnc[:, :, neuron_idx_fnc] = mean_val_fnc
            return (modified_fnc,) + args[1:]

        def frozen_norm_hook_fnc(module, input, output):
            x_fnc = input[0].float()
            inv_rms_fnc = captured_fnc["inv_rms"]
            normed_fnc = (x_fnc * inv_rms_fnc).to(module.weight.dtype) * module.weight
            return normed_fnc

        ablate_handle_fnc = down_proj_fnc.register_forward_pre_hook(ablate_hook_fnc)
        norm_freeze_handle_fnc = model_fnc.model.norm.register_forward_hook(frozen_norm_hook_fnc)

        try:
            with torch_fnc.no_grad():
                outputs_fnc = model_fnc(**inputs_fnc, use_cache=False)
        finally:
            ablate_handle_fnc.remove()
            norm_freeze_handle_fnc.remove()

        logits_fnc = outputs_fnc.logits[0, -1, :].float()
        return torch_fnc.nn.functional.softmax(logits_fnc, dim=-1)

    # 3. Load Prompts
    with open("data/contradictory_context/prompts.jsonl") as f_fnc:
        all_prompts_fnc = [json_fnc.loads(line) for line in f_fnc]

    working_prompts_fnc = [p for p in all_prompts_fnc if p["split"] == "working"]

    # 4. Define Target Layer/Neuron & Compute Mean Baseline
    layer_idx_fnc, neuron_idx_fnc = 31, 2477
    mean_val_fnc = compute_mean_activation_fnc(
        model_fnc, tokenizer_fnc, baseline_prompts, layer_idx_fnc, neuron_idx_fnc
    )

    # 5. Run Precision-Controlled Experiment Loop
    results_fnc = []

    for p_fnc in working_prompts_fnc:
        prompt_text_fnc = p_fnc["chat_formatted_prompt"]

        # Baseline forward pass with float32 logit upcasting & cache bypass
        inputs_fnc = tokenizer_fnc(prompt_text_fnc, return_tensors="pt").to(model_fnc.device)
        with torch_fnc.no_grad():
            orig_out = model_fnc(**inputs_fnc, use_cache=False)
        orig_logits = orig_out.logits[0, -1, :].float()
        orig_probs_fnc = torch_fnc.nn.functional.softmax(orig_logits, dim=-1)
        orig_entropy_fnc = compute_entropy_fnc(orig_probs_fnc)

        # Frozen norm ablation pass
        frozen_probs_fnc = frozen_norm_ablate_fnc(
            model_fnc, tokenizer_fnc, prompt_text_fnc, layer_idx_fnc, neuron_idx_fnc, mean_val_fnc
        )
        frozen_entropy_fnc = compute_entropy_fnc(frozen_probs_fnc)

        results_fnc.append({
            "prompt_id": p_fnc["prompt_id"],
            "orig_entropy": orig_entropy_fnc,
            "frozen_norm_ablated_entropy": frozen_entropy_fnc,
            "shift_under_frozen_norm": frozen_entropy_fnc - orig_entropy_fnc,
        })

    # 6. Output to CSV & Output Summary
    df_fnc = pd_fnc.DataFrame(results_fnc)
    df_fnc.to_csv("person_C_contradictory_context/results/frozen_norm_L31_N2477.csv", index=False)

    print(f"Mean shift under frozen norm: {df_fnc['shift_under_frozen_norm'].mean():.6f}")
    return


@app.cell
def _(baseline_prompts, model, tokenizer, working_prompts):
    #NEW FIXED VERSION FOR ABOVE CODE-KEEP THIS ONLY
    def _():
        import torch
        import pandas as pd
        from shared.model_utils import get_next_token_probs, compute_entropy
        from shared.ablation import compute_mean_activation

        layer_idx, neuron_idx = 31, 2477

        def frozen_norm_ablate(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

            captured = {}
            def capture_hook(module, input, output):
                x = input[0].float()
                variance = x.pow(2).mean(-1, keepdim=True)
                rms = torch.sqrt(variance + module.variance_epsilon)
                captured["rms"] = rms.detach()

            norm_handle = model.model.norm.register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(**inputs)
            norm_handle.remove()

            down_proj = model.model.layers[layer_idx].mlp.down_proj
            def ablate_hook(module, args):
                modified = args[0].clone()
                modified[:, :, neuron_idx] = mean_val
                return (modified,) + args[1:]

            def frozen_norm_hook(module, input, output):
                x = input[0].float()
                frozen_rms = captured["rms"]
                normed = (x / frozen_rms).to(module.weight.dtype) * module.weight
                return normed

            ablate_handle = down_proj.register_forward_pre_hook(ablate_hook)
            freeze_handle = model.model.norm.register_forward_hook(frozen_norm_hook)
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            finally:
                ablate_handle.remove()
                freeze_handle.remove()

            logits = outputs.logits[0, -1, :]
            return torch.nn.functional.softmax(logits, dim=-1)

        def get_probs_consistent(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            return torch.nn.functional.softmax(logits, dim=-1)

        mean_val = compute_mean_activation(model, tokenizer, baseline_prompts, layer_idx, neuron_idx)
        print("mean_val:", mean_val)

        results = []
        for p in working_prompts:
            prompt_text = p["chat_formatted_prompt"]
            orig_probs = get_probs_consistent(prompt_text)
            frozen_probs = frozen_norm_ablate(prompt_text)
            orig_entropy = compute_entropy(orig_probs)
            frozen_entropy = compute_entropy(frozen_probs)
            results.append({
                "prompt_id": p["prompt_id"],
                "orig_entropy": orig_entropy,
                "frozen_norm_ablated_entropy": frozen_entropy,
                "shift_under_frozen_norm": frozen_entropy - orig_entropy,
            })

        df = pd.DataFrame(results)
        df.to_csv("person_C_contradictory_context/results/frozen_norm_L31_N2477.csv", index=False)
        print(df["shift_under_frozen_norm"].describe())
        print(f"Mean shift under frozen norm: {df['shift_under_frozen_norm'].mean():.6f}")
        return df

    frozen_norm_result_df = _()
    return (frozen_norm_result_df,)


@app.cell
def _(frozen_norm_result_df):
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(frozen_norm_result_df["orig_entropy"], frozen_norm_result_df["frozen_norm_ablated_entropy"])
    print(f"p={p:.4f}")
    return


@app.cell
def _(baseline_prompts, model, tokenizer, working_prompts):
    #new fix- KEEP THIS ONLYY
    def _():
        import torch
        import pandas as pd
        from shared.model_utils import get_next_token_probs, compute_entropy
        from shared.ablation import compute_mean_activation

        layer_idx, neuron_idx = 31, 2477
        mean_val = compute_mean_activation(model, tokenizer, baseline_prompts, layer_idx, neuron_idx)
        print("mean_val:", mean_val)

        def frozen_norm_ablate(prompt_text):
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

            captured = {}
            def capture_hook(module, input, output):
                x = input[0].float()
                variance = x.pow(2).mean(-1, keepdim=True)
                rms = torch.sqrt(variance + module.variance_epsilon)
                captured["rms"] = rms.detach()
            norm_handle = model.model.norm.register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(**inputs)
            norm_handle.remove()

            down_proj = model.model.layers[layer_idx].mlp.down_proj

            def ablate_hook(module, args):
                modified = args[0].clone()
                modified[:, -1, neuron_idx] = mean_val   # FIXED: last position only
                return (modified,) + args[1:]

            def frozen_norm_hook(module, input, output):
                x = input[0].float()
                frozen_rms = captured["rms"]
                normed = (x / frozen_rms).to(module.weight.dtype) * module.weight
                return normed

            ablate_handle = down_proj.register_forward_pre_hook(ablate_hook)
            freeze_handle = model.model.norm.register_forward_hook(frozen_norm_hook)
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            finally:
                ablate_handle.remove()
                freeze_handle.remove()
            logits = outputs.logits[0, -1, :]
            return torch.nn.functional.softmax(logits, dim=-1)

        results = []
        for p in working_prompts:
            prompt_text = p["chat_formatted_prompt"]
            orig_probs = get_next_token_probs(model, tokenizer, prompt_text)  # reuse shared fn directly
            frozen_probs = frozen_norm_ablate(prompt_text)
            orig_entropy = compute_entropy(orig_probs)
            frozen_entropy = compute_entropy(frozen_probs)
            results.append({
                "prompt_id": p["prompt_id"],
                "orig_entropy": orig_entropy,
                "frozen_norm_ablated_entropy": frozen_entropy,
                "shift_under_frozen_norm": frozen_entropy - orig_entropy,
            })

        df = pd.DataFrame(results)
        df.to_csv("person_C_contradictory_context/results/frozen_norm_L31_N2477.csv", index=False)
        print(df["shift_under_frozen_norm"].describe())
        print(f"Mean shift under frozen norm: {df['shift_under_frozen_norm'].mean():.6f}")
        return df

    frozen_norm_result_df2 = _()
    return


@app.cell
def _(subprocess):
    check_status = subprocess.run(["git", "status"], capture_output=True, text=True)
    print(check_status.stdout)
    return


@app.cell
def _(subprocess):
    add_result = subprocess.run(
        ["git", "add",
         "person_C_contradictory_context/results/results_bf16_unquantized.csv",
         "person_C_contradictory_context/results/working_vs_heldout.csv",
         "person_C_contradictory_context/results/frozen_norm_L31_N2477.csv"],
        capture_output=True, text=True
    )
    print(add_result.stdout, add_result.stderr)
    return


@app.cell
def _(subprocess):
    identity_result1 = subprocess.run(
        ["git", "config", "--global", "user.email", "shagunchadha08@gmail.com"],
        capture_output=True, text=True
    )
    print(identity_result1.stdout, identity_result1.stderr)

    identity_result2 = subprocess.run(
        ["git", "config", "--global", "user.name", "shagunchadha"],
        capture_output=True, text=True
    )
    print(identity_result2.stdout, identity_result2.stderr)
    return


@app.cell
def _(subprocess):
    commit_result = subprocess.run(
        ["git", "commit", "-m",
         "Phase 2 remediation for contradictory_context: unquantized (bf16) ablation results, working-vs-heldout replication (0/15), and corrected frozen-norm test for L31_N2477 (fixed tokenization inconsistency and position-clamping mismatch)"],
        capture_output=True, text=True
    )
    print(commit_result.stdout, commit_result.stderr)
    return


@app.cell
def _(subprocess):
    remote_check3 = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    print(remote_check3.stdout)
    return


@app.cell
def _(subprocess):
    pull_result2 = subprocess.run(["git", "pull", "--rebase"], capture_output=True, text=True)
    print(pull_result2.stdout, pull_result2.stderr)
    return


@app.cell
def _(subprocess):
    push_result2 = subprocess.run(["git", "push"], capture_output=True, text=True)
    print(push_result2.stdout, push_result2.stderr)
    return


@app.cell
def _(subprocess):
    find_files_result = subprocess.run(
        ["find", ".", "-name", "mechanism_check_shared.json"],
        capture_output=True, text=True
    )
    print(find_files_result.stdout)
    return


@app.cell
def _(subprocess):
    rename_result = subprocess.run(
        ["git", "mv",
         "results/mechanism_check_shared.json",
         "results/mechanism_check_shared_old.json"],
        capture_output=True, text=True
    )
    print(rename_result.stdout, rename_result.stderr)
    return


@app.cell
def _(subprocess):
    move_result = subprocess.run(
        ["git", "mv",
         "mechanism_check_shared.json",
         "results/mechanism_check_shared.json"],
        capture_output=True, text=True
    )
    print(move_result.stdout, move_result.stderr)
    return


@app.cell
def _(subprocess):
    find_files_result2 = subprocess.run(
        ["find", ".", "-name", "mechanism_check_shared*.json"],
        capture_output=True, text=True
    )
    print(find_files_result2.stdout)
    return


@app.cell
def _(subprocess):
    mech_commit_result = subprocess.run(
        ["git", "commit", "-m",
         "Rename pre-audit mechanism check to mechanism_check_shared_old.json, move post-audit RMSNorm-corrected version from repo root into results/ folder"],
        capture_output=True, text=True
    )
    print(mech_commit_result.stdout, mech_commit_result.stderr)
    return


@app.cell
def _(subprocess):
    mech_pull_result = subprocess.run(["git", "pull", "--rebase"], capture_output=True, text=True)
    print(mech_pull_result.stdout, mech_pull_result.stderr)
    return


@app.cell
def _(subprocess):
    mech_push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    print(mech_push_result.stdout, mech_push_result.stderr)
    return


@app.cell
def _(subprocess):
    pull_before_next_result = subprocess.run(["git", "pull"], capture_output=True, text=True)
    print(pull_before_next_result.stdout, pull_before_next_result.stderr)
    return


@app.cell
def _(subprocess):
    guide_pull_result = subprocess.run(["git", "pull", "origin", "main"], capture_output=True, text=True)
    print(guide_pull_result.stdout, guide_pull_result.stderr)
    return


@app.cell
def _():
    path = "person_C_contradictory_context/preprocess_contradictory_context.py"

    with open(path, "r") as ff:
        content = ff.read()

    old_import = '''from person_C_contradictory_context.preprocess_contradictory_context import (
        build_template_overrides, get_field,
    )'''

    new_import = '''from person_C_contradictory_context.old_preprocess_contradictory_context import (
        build_template_overrides, get_field,
    )'''

    assert old_import in content, "old_import string not found -- check for whitespace/line-ending mismatch before proceeding"
    content = content.replace(old_import, new_import)

    with open(path, "w") as ff:
        ff.write(content)

    print("Import fixed. Confirming:")
    with open(path, "r") as ff:
        print(ff.read()[:1500])
    return


@app.cell
def _():
    inspect_path = "person_C_contradictory_context/preprocess_contradictory_context.py"

    with open(inspect_path, "r") as f_inspect:
        content_v1 = f_inspect.read()

    idx_found = content_v1.find("build_template_overrides")
    print(repr(content_v1[max(0, idx_found-200):idx_found+100]))
    return (content_v1,)


@app.cell
def _(content_v1):
    old_module_str = "person_C_contradictory_context.preprocess_contradictory_context"
    new_module_str = "person_C_contradictory_context.old_preprocess_contradictory_context"

    match_count = content_v1.count(old_module_str)
    print(f"Found {match_count} occurrence(s) of the old module path")
    return


@app.cell
def _(os):

    check_old_file = os.path.exists("person_C_contradictory_context/old_preprocess_contradictory_context.py")
    print(check_old_file)
    return


@app.cell
def _(subprocess):
    import_test_result = subprocess.run(
        ["python", "-c", "from person_C_contradictory_context.old_preprocess_contradictory_context import build_template_overrides, get_field; print('import OK')"],
        capture_output=True, text=True, cwd="."
    )
    print(import_test_result.stdout, import_test_result.stderr)
    return


@app.cell
def _(subprocess):


    status_result_c2 = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True
    )
    print("STATUS:", status_result_c2.stdout)

    diff_result_c2 = subprocess.run(
        ["git", "diff", "person_C_contradictory_context/preprocess_contradictory_context.py"],
        capture_output=True, text=True
    )
    print("DIFF:", diff_result_c2.stdout)
    return


@app.cell
def _(subprocess):
    add_result_c2 = subprocess.run(
        ["git", "add", "person_C_contradictory_context/preprocess_contradictory_context.py"],
        capture_output=True, text=True
    )
    print(add_result_c2.stdout, add_result_c2.stderr)

    commit_result_c2 = subprocess.run(
        ["git", "commit", "-m", "Fix self-import bug: point at old_preprocess_contradictory_context after rename"],
        capture_output=True, text=True
    )
    print(commit_result_c2.stdout, commit_result_c2.stderr)

    push_result_c2 = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True, text=True
    )
    print(push_result_c2.stdout, push_result_c2.stderr)
    return


@app.cell
def _(json):

    records_path_c = "data/contradictory_context/prompts.jsonl"

    records_cc = []
    with open(records_path_c, "r") as f_records:
        for line in f_records:
            records_cc.append(json.loads(line))

    print(f"Loaded {len(records_cc)} records")
    print(records_cc[0])
    return (records_cc,)


@app.cell
def _(model, records_cc, tokenizer):
    from shared.prompt_format import verify_induction_quality

    induction_result_cc = verify_induction_quality(model, tokenizer, records_cc[:25])
    print(induction_result_cc)
    return (verify_induction_quality,)


@app.cell
def _(model, records_cc, tokenizer):
    from shared.model_utils import get_next_token_probs

    inspect_n = 15
    for rec_inspect in records_cc[:inspect_n]:
        chat_prompt_inspect = rec_inspect["chat_formatted_prompt"]
        probs_inspect = get_next_token_probs(model, tokenizer, chat_prompt_inspect)
        top_id_inspect = int(probs_inspect.argmax().item())
        top_tok_inspect = tokenizer.decode([top_id_inspect])
        top_prob_inspect = float(probs_inspect.max())
        print(f"{rec_inspect['prompt_id']}: top1={top_tok_inspect!r} (p={top_prob_inspect:.3f})  |  {rec_inspect['raw_prompt'][:70]}")
    return (get_next_token_probs,)


@app.cell
def _(model, records_cc, tokenizer):
    gen_check_ids = ["cc_0000", "cc_0003", "cc_0004", "cc_0006", "cc_0012", "cc_0013", "cc_0014"]
    gen_records = [r for r in records_cc if r["prompt_id"] in gen_check_ids]

    for rec_gen in gen_records:
        inputs_gen = tokenizer(rec_gen["chat_formatted_prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
        output_gen = model.generate(**inputs_gen, max_new_tokens=15, do_sample=False)
        generated_text_gen = tokenizer.decode(output_gen[0][inputs_gen["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{rec_gen['prompt_id']}: {rec_gen['raw_prompt'][:60]!r}")
        print(f"   -> {generated_text_gen!r}")
        print()
    return


@app.cell
def _():
    final_file_path_c = "person_C_contradictory_context/preprocess_contradictory_context.py"

    final_file_content_c = '''"""
    preprocess_contradictory_context_v2.py -- Person C, Phase 3

    Two changes from preprocess_contradictory_context.py:

    1. RESTORE MODEL-KNOWS FILTER: the live pipeline (preprocess_contradictory_context.py)
       never checked that the model actually has parametric knowledge of target_true
       before building a contradiction prompt around it -- for an obscure CounterFact
       subject, the prompt then tests "the model never knew this" (lack-of-knowledge)
       dressed up as a contradiction test, not genuine conflict-resolution uncertainty.

       NOTE ON A BUG IN THE FIRST VERSION OF THIS FILE: the check was initially ported
       directly from build_dataset.py's model_knows_fact(), which only tests
       top1_prob > threshold and never looks at WHICH token that is -- true_object was
       accepted as a parameter and never used. That passes a model that's confidently
       wrong just as easily as one that's confidently right, which is not "requiring
       top token = true object" as the remediation plan actually specifies. Fixed
       below: the top-1 token is decoded and compared against true_object's own first
       token before the fact counts as "known." The probability threshold is now a
       secondary, optional strictness knob on top of that match, not a substitute for it.

       NOTE: this makes preprocessing model-dependent -- unlike the current CPU-only
       preprocess_contradictory_context.py, this needs `model` + `tokenizer` already
       loaded (same pattern as build_dataset.py). Run this on Kaggle/RunPod, not locally.

    2. MATCHED CONTROLS: same "Redefine" surface structure, but reasserting the TRUE
       fact instead of a false one -- "Redefine: Thailand belongs to the continent of
       Asia. Thailand belongs to the continent of" (context confirms what the model
       already believes, so there's no conflict to resolve). Same subjects/relations
       as the working set where possible, so the control is a genuine twin, not just
       a different sample.

    3. TURN-BOUNDARY FIX (added after verify_induction_quality + generation inspection
       showed this category has the SAME bug as A/B, contrary to shared/prompt_format.py's
       original assumption): the "Redefine: ... {base_prompt}" construction was ending
       the USER turn on the bare relation phrase, then opening a FRESH assistant turn --
       model was generating turn-openers ("I couldn't find any information...", "I think
       there may be a mistake...") instead of continuing the sentence. Fixed by routing
       through build_completion_prompt / build_records_with_formatter so base_prompt is
       PREFILLED into the assistant turn instead of left as an open new turn.

    Run from repo root, inside a Kaggle/RunPod session with model+tokenizer already
    loaded (same convention as build_dataset.py):
        exec(open("person_C_contradictory_context/preprocess_contradictory_context.py").read())
        build_and_save(model, tokenizer)
    """

    import sys
    sys.path.append(".")
    import json
    from datasets import load_dataset

    from person_C_contradictory_context.old_preprocess_contradictory_context import (
        build_template_overrides, get_field,
    )
    from shared.model_utils import get_next_token_probs, compute_top1_prob
    from shared.prompt_format import build_completion_prompt, build_records_with_formatter

    OUTPUT_PATH = "data/contradictory_context/prompts.jsonl"
    CONTROLS_OUTPUT_PATH = "data/contradictory_context/controls.jsonl"


    def model_knows_fact(
        model, tokenizer, base_prompt: str, true_object: str,
        min_prob: float = None, verbose: bool = False,
    ) -> bool:
        """
        Parametric-knowledge check, done on base_prompt BEFORE the contradiction
        context is layered on top: requires the model's TOP-1 next token to
        match true_object's own first token -- not just "the model is
        confident about something."

        true_object is often multi-token/multi-word ("United Kingdom", "Marie
        Curie"). Rather than requiring the whole phrase to match in one token
        (which would reject almost everything), this follows the standard
        ROME/CounterFact convention: tokenize true_object on its own (with a
        leading space, matching how it would actually continue the prompt) and
        compare the model's top-1 token against ONLY true_object's first token.
        This is a heuristic, not a proof the model "knows" the full fact --
        pilot on ~20 examples with verbose=True and read the printed
        top-token/true-object pairs before trusting it at scale, same as you'd
        do for any new filter in this project.

        min_prob is now an OPTIONAL secondary gate (default off) on top of the
        match requirement, for cases where you want to additionally require a
        minimum confidence, not a substitute for checking token identity.
        """
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": base_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        probs = get_next_token_probs(model, tokenizer, chat_prompt)
        top1_prob = compute_top1_prob(probs)
        top_token_id = int(probs.argmax().item())
        top_token_str = tokenizer.decode([top_token_id]).strip().lower()

        true_object_token_ids = tokenizer.encode(" " + true_object.strip(), add_special_tokens=False)
        if not true_object_token_ids:
            return False
        true_object_first_token_str = tokenizer.decode([true_object_token_ids[0]]).strip().lower()

        is_match = top_token_str == true_object_first_token_str
        passes_prob_gate = (min_prob is None) or (top1_prob > min_prob)

        if verbose:
            match_flag = "MATCH" if is_match else "no match"
            print(f"    top1={top_token_str!r} (p={top1_prob:.3f})  vs  "
                  f"true_object first token={true_object_first_token_str!r}  [{match_flag}]")

        return is_match and passes_prob_gate


    def extract_fields_flexible(record, unresolved: set):
        """Same field-extraction logic as preprocess_contradictory_context.py's
        extract_fields(), duplicated here to avoid a fragile cross-import of a
        "private" helper -- keep in sync if that file's schema handling changes."""
        subject = get_field(record, ("requested_rewrite", "subject"), ("subject",))
        prompt_template = get_field(record, ("requested_rewrite", "prompt"), ("prompt",))
        target_true = get_field(record, ("requested_rewrite", "target_true", "str"), ("target_true",))
        target_new = get_field(record, ("requested_rewrite", "target_new", "str"), ("target_new",))
        relation_id = get_field(record, ("requested_rewrite", "relation_id"), ("relation_id",))

        if not all([subject, prompt_template, target_true, target_new]):
            return None
        if target_true.strip().lower() == target_new.strip().lower():
            return None
        if relation_id in unresolved:
            return None

        return {
            "subject": subject, "prompt_template": prompt_template,
            "target_true": target_true.strip(), "target_new": target_new.strip(),
            "relation_id": relation_id,
        }


    def _redefine_formatter(tokenizer, raw_prompt, **kwargs):
        """
        raw_prompt arrives as 'Redefine: {base_prompt} {target}.|||SPLIT|||{base_prompt}'.
        Splits into the user-turn content (the full Redefine sentence) and the
        continuation (bare base_prompt) that must be PREFILLED into the assistant
        turn, not left as a fresh open turn -- this is the fix for the
        turn-boundary bug confirmed via verify_induction_quality + generation
        inspection (model was producing "I couldn't find any information..." /
        "I think there may be a mistake..." instead of continuing the sentence).
        """
        user_content, continuation = raw_prompt.split("|||SPLIT|||")
        formatted = build_completion_prompt(
            tokenizer, user_content, cloze_suffix=continuation, ensure_question_mark=False,
        )
        formatted["raw_prompt"] = f"{user_content} {continuation}"
        return formatted


    def build_and_save(model, tokenizer, n_target=120, knows_fact_min_prob=None, verbose_knows_fact=False, seed=42):
        print("Building template overrides from ParaRel...")
        overrides, unresolved = build_template_overrides()
        print(f"{len(overrides)} relations auto-patched; {len(unresolved)} unresolved (dropped)")

        print("\\nLoading azhx/counterfact ...")
        ds = load_dataset("azhx/counterfact", split="train")

        contradiction_raw, control_raw = [], []
        checked, knows_fact_count = 0, 0

        for record in ds:
            fields = extract_fields_flexible(record, unresolved)
            if fields is None:
                continue
            template = overrides.get(fields["relation_id"], fields["prompt_template"])
            base_prompt = template.format(fields["subject"]).strip()

            checked += 1
            if not model_knows_fact(model, tokenizer, base_prompt, fields["target_true"],
                                     min_prob=knows_fact_min_prob, verbose=verbose_knows_fact):
                continue
            knows_fact_count += 1

            contradiction_raw.append(
                f"Redefine: {base_prompt} {fields['target_new']}.|||SPLIT|||{base_prompt}"
            )
            control_raw.append(
                f"Redefine: {base_prompt} {fields['target_true']}.|||SPLIT|||{base_prompt}"
            )

            if len(contradiction_raw) >= n_target * 2:  # headroom before subsampling in build_records
                break

        print(f"Checked {checked} rows; model knew the true fact for {knows_fact_count} "
              f"({knows_fact_count/max(checked,1)*100:.1f}%) -- these are the only ones "
              f"used, for both the contradiction set and its matched true-object control.")

        records = build_records_with_formatter(
            raw_prompts=contradiction_raw,
            category="contradictory_context",
            source_dataset="CounterFact + ParaRel, model-knows-filtered",
            prefix="cc",
            tokenizer=tokenizer,
            formatter=_redefine_formatter,
            n_working=n_target,
            split_ratio=0.7,
            seed=seed,
        )
        with open(OUTPUT_PATH, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\\n")
        print(f"Saved {len(records)} contradiction prompts to {OUTPUT_PATH}")

        control_records = build_records_with_formatter(
            raw_prompts=control_raw,
            category="contradictory_context",
            source_dataset="CounterFact + ParaRel, true-object control (matched)",
            prefix="cc_ctrl",
            tokenizer=tokenizer,
            formatter=_redefine_formatter,
            n_working=n_target,
            split_ratio=0.7,
            seed=seed,  # SAME seed + same underlying subject/relation ordering as
                        # the contradiction set above, so cc_ctrl_0007 and cc_0007
                        # are the SAME subject/relation -- a true matched pair.
            is_control=True,
        )
        with open(CONTROLS_OUTPUT_PATH, "w") as f:
            for r in control_records:
                f.write(json.dumps(r) + "\\n")
        print(f"Saved {len(control_records)} matched true-object control prompts to {CONTROLS_OUTPUT_PATH}")

        return records, control_records


    if __name__ == "__main__":
        print("This script expects `model` and `tokenizer` already loaded in your "
              "session (same convention as build_dataset.py) -- call "
              "build_and_save(model, tokenizer) directly rather than running this "
              "file standalone.")
    '''

    with open(final_file_path_c, "w") as f_final_write:
        f_final_write.write(final_file_content_c)

    print("File written. Length:", len(final_file_content_c))
    return


@app.cell
def _(subprocess):
    diff_result_c3 = subprocess.run(
        ["git", "diff", "person_C_contradictory_context/preprocess_contradictory_context.py"],
        capture_output=True, text=True
    )
    print(diff_result_c3.stdout)
    return


@app.cell
def _():
    exec(open("person_C_contradictory_context/preprocess_contradictory_context.py").read())
    print("build_and_save now defined:", "build_and_save" in dir())
    redefine_formatter_public = globals()['_redefine_formatter']
    print("redefine_formatter_public bound:", redefine_formatter_public)
    return (redefine_formatter_public,)


@app.cell
def _(build_and_save, model, tokenizer):
    records_rebuilt_c, control_records_rebuilt_c = build_and_save(model, tokenizer)
    print(f"Rebuilt {len(records_rebuilt_c)} working records, {len(control_records_rebuilt_c)} control records")
    return


@app.cell
def _(json, model, tokenizer, verify_induction_quality):
    records_path_c_v3 = "data/contradictory_context/prompts.jsonl"
    records_cc_v3 = []
    with open(records_path_c_v3, "r") as f_records_v3:
        for line_v3 in f_records_v3:
            records_cc_v3.append(json.loads(line_v3))

    induction_result_cc_v3 = verify_induction_quality(model, tokenizer, records_cc_v3[:25])
    print(induction_result_cc_v3)
    return (records_cc_v3,)


@app.cell
def _(model, records_cc_v3, tokenizer):
    peaked_check_records_v3 = [r for r in records_cc_v3[:25] if r["prompt_id"] in
        ["cc_0002", "cc_0005", "cc_0007", "cc_0011", "cc_0015", "cc_0019", "cc_0022"]]

    for rec_peak_v3 in peaked_check_records_v3:
        inputs_peak_v3 = tokenizer(rec_peak_v3["chat_formatted_prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
        output_peak_v3 = model.generate(**inputs_peak_v3, max_new_tokens=15, do_sample=False)
        generated_peak_v3 = tokenizer.decode(output_peak_v3[0][inputs_peak_v3["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{rec_peak_v3['prompt_id']}: {rec_peak_v3['raw_prompt'][:70]!r}")
        print(f"   -> {generated_peak_v3!r}")
        print()
    return


@app.cell
def _(json, model, tokenizer, verify_induction_quality):
    controls_path_c_v3 = "data/contradictory_context/controls.jsonl"
    control_records_cc_v3 = []
    with open(controls_path_c_v3, "r") as f_controls_v3:
        for line_ctrl_v3 in f_controls_v3:
            control_records_cc_v3.append(json.loads(line_ctrl_v3))

    induction_result_cc_controls_v3 = verify_induction_quality(model, tokenizer, control_records_cc_v3[:25])
    print(induction_result_cc_controls_v3)
    return


@app.cell
def _(compute_top1_prob, get_next_token_probs):
    #checking through pilot if this can be done for 120 or not to fix the limitation : answer: its pointless
    def model_knows_fact_bounded(
        model, tokenizer, base_prompt: str, true_object: str,
        min_prob: float = 0.3, max_prob: float = 0.85, verbose: bool = False,
    ) -> bool:
        """Pilot variant: requires top-1 match AND moderate confidence (min_prob < p < max_prob),
        testing whether excluding near-certain facts produces a real contradiction-vs-control gap."""
        chat_prompt_bound = tokenizer.apply_chat_template(
            [{"role": "user", "content": base_prompt}], tokenize=False, add_generation_prompt=True,
        )
        probs_bound = get_next_token_probs(model, tokenizer, chat_prompt_bound)
        top1_prob_bound = compute_top1_prob(probs_bound)
        top_token_id_bound = int(probs_bound.argmax().item())
        top_token_str_bound = tokenizer.decode([top_token_id_bound]).strip().lower()
        true_object_token_ids_bound = tokenizer.encode(" " + true_object.strip(), add_special_tokens=False)
        if not true_object_token_ids_bound:
            return False
        true_object_first_token_str_bound = tokenizer.decode([true_object_token_ids_bound[0]]).strip().lower()
        is_match_bound = top_token_str_bound == true_object_first_token_str_bound
        in_band_bound = min_prob < top1_prob_bound < max_prob
        if verbose:
            print(f"    top1={top_token_str_bound!r} (p={top1_prob_bound:.3f})  match={is_match_bound}  in_band={in_band_bound}")
        return is_match_bound and in_band_bound

    print("model_knows_fact_bounded defined")
    return (model_knows_fact_bounded,)


@app.cell
def _(
    build_template_overrides,
    extract_fields_flexible,
    load_dataset,
    model,
    model_knows_fact_bounded,
    tokenizer,
):
    print("Building template overrides from ParaRel (pilot)...")
    overrides_pilot, unresolved_pilot = build_template_overrides()

    print("Loading azhx/counterfact (pilot, reusing cached dataset)...")
    ds_pilot = load_dataset("azhx/counterfact", split="train")

    contradiction_raw_pilot, control_raw_pilot = [], []
    checked_pilot, knows_fact_count_pilot = 0, 0
    n_target_pilot = 15

    for record_pilot in ds_pilot:
        fields_pilot = extract_fields_flexible(record_pilot, unresolved_pilot)
        if fields_pilot is None:
            continue
        template_pilot = overrides_pilot.get(fields_pilot["relation_id"], fields_pilot["prompt_template"])
        base_prompt_pilot = template_pilot.format(fields_pilot["subject"]).strip()

        checked_pilot += 1
        if not model_knows_fact_bounded(model, tokenizer, base_prompt_pilot, fields_pilot["target_true"]):
            continue
        knows_fact_count_pilot += 1

        contradiction_raw_pilot.append(
            f"Redefine: {base_prompt_pilot} {fields_pilot['target_new']}.|||SPLIT|||{base_prompt_pilot}"
        )
        control_raw_pilot.append(
            f"Redefine: {base_prompt_pilot} {fields_pilot['target_true']}.|||SPLIT|||{base_prompt_pilot}"
        )

        if len(contradiction_raw_pilot) >= n_target_pilot:
            break

    print(f"Checked {checked_pilot} rows; {knows_fact_count_pilot} passed the bounded filter "
          f"({knows_fact_count_pilot/max(checked_pilot,1)*100:.1f}%)")
    return contradiction_raw_pilot, control_raw_pilot


@app.cell
def _(
    build_records_with_formatter,
    contradiction_raw_pilot,
    control_raw_pilot,
    model,
    redefine_formatter_public,
    tokenizer,
    verify_induction_quality,
):
    pilot_records = build_records_with_formatter(
        raw_prompts=contradiction_raw_pilot,
        category="contradictory_context",
        source_dataset="pilot-bounded",
        prefix="cc_pilot",
        tokenizer=tokenizer,
        formatter=redefine_formatter_public,
        n_working=len(contradiction_raw_pilot),
        split_ratio=1.0,
        seed=42,
    )

    pilot_control_records = build_records_with_formatter(
        raw_prompts=control_raw_pilot,
        category="contradictory_context",
        source_dataset="pilot-bounded-control",
        prefix="cc_pilot_ctrl",
        tokenizer=tokenizer,
        formatter=redefine_formatter_public,
        n_working=len(control_raw_pilot),
        split_ratio=1.0,
        seed=42,
        is_control=True,
    )

    print("Pilot working prompts:")
    induction_result_pilot = verify_induction_quality(model, tokenizer, pilot_records)
    print(induction_result_pilot)

    print("\nPilot control prompts:")
    induction_result_pilot_control = verify_induction_quality(model, tokenizer, pilot_control_records)
    print(induction_result_pilot_control)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compact version:

    Contradictory-context prompts do not reliably induce measurable uncertainty relative to matched true-fact controls. Across both an unrestricted sample (n=25, working mean top1=0.850 vs. control mean top1=0.804) and a pilot restricted to moderate-confidence base facts (top1 between 0.3–0.85; n=15, working mean=0.784 vs. control mean=0.804), the contradiction condition showed no consistent gap from its no-conflict control — in several matched pairs the contradiction prompt was more confidently resolved than the control. This suggests the instruct-tuned model tends to correct injected false context confidently rather than holding genuine split-uncertainty between the two competing claims, at least for the well-known CounterFact-style facts that pass the parametric-knowledge filter. Restricting to moderate-confidence base facts did not resolve this, indicating the effect is not primarily driven by facts being too well-known to begin with — this appears to be a more fundamental property of how the model resolves single-turn factual contradictions, consistent with Context Copying Modulation (2025)'s finding that context-conflict resolution in this model family doesn't map cleanly onto the same entropy-neuron mechanisms as ambiguity/knowledge-absence uncertainty.

    Fuller version, with more explicit methodology (better for an appendix/methods section):

    Contradictory context does not show a clean uncertainty-vs-control gap. The category's construction (a "Redefine: [false/true fact]. [base_prompt]" prompt, with the assistant turn prefilled to genuinely continue the sentence rather than open a fresh chat turn) was verified to resolve the turn-boundary artifact present in an earlier version — generation inspection confirmed the model completes the sentence directly (e.g., "Adobe Flash is created by ➝ Adobe Systems, not Apple") rather than producing disclaiming preamble ("I couldn't find any information...").

    However, once that artifact was fixed, a second and more fundamental issue emerged: the model resolves the injected contradiction with roughly the same confidence it shows when simply restating the true fact with no conflict present at all. On a 25-item sample, working (contradiction) prompts averaged top1=0.850 versus 0.804 for matched true-object controls — no meaningful separation, and directionally the wrong way (contradiction more confident than control). We hypothesized this might be driven by the parametric-knowledge filter selecting only very well-known facts (major tech companies, common geography), where the model's prior is strong enough that a single contradicting sentence produces little genuine conflict. To test this, we piloted a bounded variant of the knowledge filter requiring moderate rather than maximal base-fact confidence (top1 between 0.3 and 0.85). This did not produce a gap either (n=15: working mean=0.784, control mean=0.804) — matched pairs such as "Shablykinsky District... Belarus" (0.959) vs. its true-fact control (0.996) show the same pattern at smaller scale.

    We treat this as a genuine finding rather than a bug to engineer around: for this model and this contradiction construction, factual correction appears to happen via confident override rather than sustained distributional uncertainty, at least for CounterFact-style single-fact conflicts. This is consistent with the category's original "hardest to get clean causal results from" ranking (Section 3) and with Context Copying Modulation (2025)'s finding that Llama-3-8B's context-conflict behavior doesn't transfer cleanly from GPT-2-style entropy neuron signatures. Any downstream causal results (Phase 3/4) for this category should be interpreted with this limitation in mind — a null or weak result for contradictory-context neurons may reflect genuinely weak induced uncertainty in the dataset rather than absence of a shared mechanism.
    """)
    return


@app.cell
def _(subprocess):
    status_result_final = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True
    )
    print(status_result_final.stdout)
    return


@app.cell
def _():
    limitations_path_c = "person_C_contradictory_context/LIMITATIONS.md"

    limitations_content_c = """# Contradictory Context — Known Limitation

    ## Contradictory context does not show a clean uncertainty-vs-control gap

    The category's construction ("Redefine: [false/true fact]. [base_prompt]", with the
    assistant turn prefilled to genuinely continue the sentence rather than open a fresh
    chat turn) was verified to resolve the turn-boundary artifact present in an earlier
    version -- generation inspection confirmed the model completes the sentence directly
    (e.g., "Adobe Flash is created by" -> "Adobe Systems, not Apple") rather than producing
    disclaiming preamble ("I couldn't find any information...").

    However, once that artifact was fixed, a second and more fundamental issue emerged:
    the model resolves the injected contradiction with roughly the same confidence it shows
    when simply restating the true fact with no conflict present at all. On a 25-item sample,
    working (contradiction) prompts averaged top1=0.850 versus 0.804 for matched true-object
    controls -- no meaningful separation, and directionally the wrong way (contradiction more
    confident than control).

    We hypothesized this might be driven by the parametric-knowledge filter selecting only
    very well-known facts (major tech companies, common geography), where the model's prior
    is strong enough that a single contradicting sentence produces little genuine conflict.
    To test this, we piloted a bounded variant of the knowledge filter requiring moderate
    rather than maximal base-fact confidence (top1 between 0.3 and 0.85). This did not
    produce a gap either (n=15: working mean=0.784, control mean=0.804) -- matched pairs such
    as "Shablykinsky District... Belarus" (0.959) vs. its true-fact control (0.996) show the
    same pattern at smaller scale.

    We treat this as a genuine finding rather than a bug to engineer around: for this model
    and this contradiction construction, factual correction appears to happen via confident
    override rather than sustained distributional uncertainty, at least for CounterFact-style
    single-fact conflicts. This is consistent with the category's original "hardest to get
    clean causal results from" ranking and with Context Copying Modulation (2025)'s finding
    that Llama-3-8B's context-conflict behavior doesn't transfer cleanly from GPT-2-style
    entropy neuron signatures. Any downstream causal results (Phase 3/4) for this category
    should be interpreted with this limitation in mind -- a null or weak result for
    contradictory-context neurons may reflect genuinely weak induced uncertainty in the
    dataset rather than absence of a shared mechanism.
    """

    with open(limitations_path_c, "w") as f_limitations:
        f_limitations.write(limitations_content_c)

    print("Limitations doc written.")
    return


@app.cell
def _(subprocess):
    status_check_final = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True
    )
    print(status_check_final.stdout)
    return


@app.cell
def _(subprocess):
    add_result_all = subprocess.run(
        ["git", "add",
         "person_C_contradictory_context/preprocess_contradictory_context.py",
         "data/contradictory_context/prompts.jsonl",
         "data/contradictory_context/controls.jsonl",
         "person_C_contradictory_context/LIMITATIONS.md"],
        capture_output=True, text=True
    )
    print(add_result_all.stdout, add_result_all.stderr)

    commit_result_all = subprocess.run(
        ["git", "commit", "-m",
         "Fix turn-boundary bug in contradictory-context prompts (prefill assistant turn); "
         "document confirmed limitation that contradiction vs control shows no reliable "
         "confidence gap even after this fix and a bounded-confidence pilot"],
        capture_output=True, text=True
    )
    print(commit_result_all.stdout, commit_result_all.stderr)

    push_result_all = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True, text=True
    )
    print(push_result_all.stdout, push_result_all.stderr)
    return


@app.cell
def _():
    def cleanup_nested():
        import shutil
        import subprocess
        from pathlib import Path

        nested_folder = Path("Confidence-Neurons-Across-Uncertainty-Types")
        if nested_folder.exists() and nested_folder.is_dir():
            shutil.rmtree(nested_folder)
            print("Removed leftover nested folder!")

        print("\n" + subprocess.check_output(["git", "status"]).decode())


    cleanup_nested()
    return


@app.cell
def _():
    def check_last_commit():
        import subprocess

        # Shows the commit hash, author, date, and message of the last commit
        output = subprocess.check_output(
            ["git", "log", "-1", "--stat"]
        ).decode()
        print(output)


    check_last_commit()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
