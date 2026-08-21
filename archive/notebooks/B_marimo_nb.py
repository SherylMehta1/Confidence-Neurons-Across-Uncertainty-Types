import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import subprocess
    from pathlib import Path
    import shutil

    # 1. Check for tokens in environment
    HF_TOKEN_2 = os.environ.get("HF_TOKEN")
    GH_TOKEN = os.environ.get("GITHUB_TOKEN")

    # 2. Build the correct, valid Git URL
    repo_user = "SherylMehta1"
    repo_name = "Confidence-Neurons-Across-Uncertainty-Types"

    if GH_TOKEN and GH_TOKEN.strip():
        clean_token = GH_TOKEN.strip()
        remote_url = f"https://{clean_token}@github.com/{repo_user}/{repo_name}.git"
        print("Secrets loaded: Authenticated Git URL configured.")
    else:
        remote_url = f"https://github.com/{repo_user}/{repo_name}.git"
        print("GITHUB_TOKEN not found in os.environ. Using standard public URL.")
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
            shutil.rmtree(repo_path)
        subprocess.run(["git", "clone", remote_url], check=True)
        os.chdir(repo_path)
        print("Repository cloned successfully.")

    print(f"Current Working Directory: {Path.cwd()}")
    return Path, os, shutil, subprocess


@app.cell
def _(os):
    #import os

    HF_TOKEN = os.environ["HF_TOKEN"]
    return (HF_TOKEN,)


@app.cell
def _(HF_TOKEN):
    HF_TOKEN is not None
    return


@app.cell
def _(HF_TOKEN):
    from huggingface_hub import login

    login(token=HF_TOKEN)
    return


@app.cell
def _():
    #import subprocess

    #subprocess.run(
    #    [
    #        "git",
     #       "clone",
      #      "https://github.com/SherylMehta1/Confidence-Neurons-Across-Uncertainty-Types.git",
       # ],
        #check=True,
    #)
    return


@app.cell
def _():
    #from pathlib import Path

    #repo = Path("Confidence-Neurons-Across-Uncertainty-Types")
    #os.chdir(repo)

    #print(Path.cwd())
    return


@app.cell
def _(subprocess):
    subprocess.run(["git", "pull"], check=True)
    return


@app.cell
def _(subprocess):
    import sys

    subprocess.run(
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
        ],
        check=True,
    )
    return (sys,)


@app.cell
def _(HF_TOKEN):
    from huggingface_hub import whoami

    info = whoami(token=HF_TOKEN)

    print("Logged in as:", info["name"])
    return


@app.cell
def _(sys):
    sys.path.append(".")
    from shared.model_utils import load_model
    model, tokenizer = load_model(quantize=True)
    return model, tokenizer


@app.cell
def _():
    CATEGORY = "lack_of_knowledge"   # or "ambiguity" / "contradictory_context"
    DATA_PATH = f"data/{CATEGORY}/prompts.jsonl"
    RESULTS_PATH = f"person_B_lack_of_knowledge/results/results.csv"  # adjust folder per person
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
def _(subprocess):
    result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(result.stdout or "Working tree clean")
    return (result,)


@app.cell
def _(subprocess):
    subprocess.run(["git", "stash", "push", "-u", "-m", "marimo research work"], check=True)
    return


@app.cell
def _(subprocess):
    subprocess.run(["git", "pull", "--rebase"], check=True)
    return


@app.cell
def _(subprocess):
    subprocess.run(["git", "stash", "pop"], check=True)
    return


@app.cell
def _(subprocess):
    result2 = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(result2.stdout)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        ["git", "add", "person_B_lack_of_knowledge/results/results.csv"],
        check=True
    )
    return


@app.cell
def _(os, subprocess):
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

    subprocess.run(["git", "config", "--global", "user.email", "sherylmehta11@gmail.com"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "SherylMehta1"], check=True)
    subprocess.run(
        ["git", "remote", "set-url", "origin",
         f"https://{GITHUB_TOKEN}@github.com/SherylMehta1/Confidence-Neurons-Across-Uncertainty-Types.git"],
        check=True,
    )
    return


@app.cell
def _(subprocess):
    subprocess.run(
        ["git", "commit", "-m", "Update results with fixed Phase 1 pipeline for lack of knowledge"],
        check=True
    )
    return


@app.cell
def _(subprocess):
    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    print(push_result.stdout, push_result.stderr)
    return


@app.cell
def _(subprocess):
    subprocess.run(["git", "pull", "--rebase"], check=True)
    return


@app.cell
def _():
    # PHASE 2
    return


@app.cell
def _(baseline_prompts):
    # Phase 2: unquantized bf16 rerun -- lack_of_knowledge
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
        "person_B_lack_of_knowledge/results/significance_summary.csv"
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
    with open("data/lack_of_knowledge/prompts.jsonl") as f_prompts_p2:
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
                category="lack_of_knowledge",
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
        "person_B_lack_of_knowledge/results/results_bf16_unquantized.csv",
        index=False,
    )
    print(
        f"Saved {len(df_bf16_p2)} rows to "
        "person_B_lack_of_knowledge/results/results_bf16_unquantized.csv"
    )
    return


@app.cell
def _(baseline_prompts):
    # Phase 2: held-out evaluation -- lack_of_knowledge
    import json as json_p2hb
    import sys as sys_p2hb
    sys_p2hb.path.append(".")
    import pandas as pd_p2hb
    from scipy.stats import wilcoxon as wilcoxon_p2hb

    from shared.model_utils import load_model as load_model_p2hb
    from shared.ablation import (
        compute_mean_activation as compute_mean_activation_p2hb,
        run_ablation_experiment as run_ablation_experiment_p2hb,
    )

    model_p2hb, tokenizer_p2hb = load_model_p2hb(quantize=False)

    with open("candidate_neurons.json") as f_p2hb:
        candidates_p2hb = json_p2hb.load(f_p2hb)
    if isinstance(candidates_p2hb[0], list):
        candidates_p2hb = [
            {
                "layer": c[0],
                "neuron_idx": c[1],
                "neuron_id": f"L{c[0]}_N{c[1]}",
            }
            for c in candidates_p2hb
        ]

    with open("data/lack_of_knowledge/prompts.jsonl") as f_prompts_p2hb:
        all_prompts_p2hb = [
            json_p2hb.loads(line)
            for line in f_prompts_p2hb
        ]

    working_prompts_p2hb = [
        p
        for p in all_prompts_p2hb
        if p["split"] == "working"
    ]
    held_out_prompts_p2hb = [
        p
        for p in all_prompts_p2hb
        if p["split"] == "held_out"
    ]

    assert len(held_out_prompts_p2hb) == 36, (
        f"Expected 36 held-out prompts, got {len(held_out_prompts_p2hb)} -- "
        f"held-out filter may still be broken"
    )

    comparison_rows_p2hb = []
    for candidate_p2hb in candidates_p2hb:
        layer_idx_p2hb = candidate_p2hb["layer"]
        neuron_idx_p2hb = candidate_p2hb["neuron_idx"]
        mean_val_p2hb = compute_mean_activation_p2hb(
            model_p2hb,
            tokenizer_p2hb,
            baseline_prompts,
            layer_idx_p2hb,
            neuron_idx_p2hb,
        )

        working_rows_p2hb = run_ablation_experiment_p2hb(
            model_p2hb,
            tokenizer_p2hb,
            working_prompts_p2hb,
            layer_idx_p2hb,
            neuron_idx_p2hb,
            mean_val_p2hb,
            category="lack_of_knowledge",
            split="working",
        )
        heldout_rows_p2hb = run_ablation_experiment_p2hb(
            model_p2hb,
            tokenizer_p2hb,
            held_out_prompts_p2hb,
            layer_idx_p2hb,
            neuron_idx_p2hb,
            mean_val_p2hb,
            category="lack_of_knowledge",
            split="held_out",
        )

        w_df_p2hb = pd_p2hb.DataFrame(working_rows_p2hb)
        h_df_p2hb = pd_p2hb.DataFrame(heldout_rows_p2hb)

        w_stat_p2hb, w_p_p2hb = wilcoxon_p2hb(
            w_df_p2hb["orig_entropy"], w_df_p2hb["ablated_entropy"]
        )
        h_stat_p2hb, h_p_p2hb = wilcoxon_p2hb(
            h_df_p2hb["orig_entropy"], h_df_p2hb["ablated_entropy"]
        )

        comparison_rows_p2hb.append({
            "neuron_id": candidate_p2hb["neuron_id"],
            "working_mean_shift": w_df_p2hb["entropy_shift"].mean(),
            "working_p": w_p_p2hb,
            "working_n": len(w_df_p2hb),
            "heldout_mean_shift": h_df_p2hb["entropy_shift"].mean(),
            "heldout_p": h_p_p2hb,
            "heldout_n": len(h_df_p2hb),
            "replicates": (w_p_p2hb < 0.01) and (h_p_p2hb < 0.01),
        })
        print(
            f"{candidate_p2hb['neuron_id']}: working p={w_p_p2hb:.4f}, "
            f"held-out p={h_p_p2hb:.4f}, "
            f"replicates={comparison_rows_p2hb[-1]['replicates']}"
        )

    comparison_df_p2hb = pd_p2hb.DataFrame(comparison_rows_p2hb)
    comparison_df_p2hb.to_csv(
        "person_B_lack_of_knowledge/results/working_vs_heldout.csv",
        index=False,
    )
    print(
        f"\n{comparison_df_p2hb['replicates'].sum()} / {len(comparison_df_p2hb)} "
        "candidates replicate on held-out"
    )
    return


@app.cell
def _():
    # Phase 2: Stolfo weight criteria -- weights only, CPU, run ONCE
    # (candidate_neurons.json is shared/frozen across all 3 categories,
    # so this should not be duplicated by B and C -- confirm before committing)
    import json as json_stf
    import sys as sys_stf
    sys_stf.path.append(".")
    import random as random_stf
    import numpy as np_stf
    import pandas as pd_stf
    import torch as torch_stf

    from shared.model_utils import load_model as load_model_stf

    model_stf, tokenizer_stf = load_model_stf(quantize=False)
    model_stf.eval()

    # ------------------------------------------------------------
    # 1. Load candidates
    # ------------------------------------------------------------
    with open("candidate_neurons.json") as f_stf:
        candidates_stf = json_stf.load(f_stf)
    if isinstance(candidates_stf[0], list):
        candidates_stf = [
            {
                "layer": c[0],
                "neuron_idx": c[1],
                "neuron_id": f"L{c[0]}_N{c[1]}",
            }
            for c in candidates_stf
        ]
    print(f"Loaded {len(candidates_stf)} candidates")

    # ------------------------------------------------------------
    # 2. SVD of the unembedding matrix -- computed ONCE, reused for
    #    every neuron (candidate or random)
    # ------------------------------------------------------------
    W_U_stf = model_stf.get_output_embeddings().weight.detach().float()  # [vocab, hidden]
    print(f"W_U shape: {W_U_stf.shape}")

    U_stf, S_stf, Vt_stf = torch_stf.linalg.svd(W_U_stf, full_matrices=False)
    print(f"Singular value range: {S_stf.min().item():.4f} to {S_stf.max().item():.4f}")

    # Null-space proxy: bottom 10% of singular-value directions -- a
    # chosen cutoff, not a given; verify against Stolfo et al. before
    # treating as final
    NULL_FRACTION_CUTOFF_STF = 0.10
    n_null_dims_stf = max(1, int(len(S_stf) * NULL_FRACTION_CUTOFF_STF))
    null_space_basis_stf = Vt_stf[-n_null_dims_stf:, :]
    print(f"Using bottom {n_null_dims_stf} of {len(S_stf)} singular directions as null-space proxy")

    # ------------------------------------------------------------
    # 3. RMSNorm gamma (final norm, applied before unembedding)
    # ------------------------------------------------------------
    final_norm_gamma_stf = model_stf.model.norm.weight.detach().float()  # [hidden]

    # ------------------------------------------------------------
    # 4. Per-neuron Stolfo criteria
    # ------------------------------------------------------------
    def compute_stolfo_criteria_stf(layer_idx_stf, neuron_idx_stf):
        down_proj_stf = model_stf.model.layers[layer_idx_stf].mlp.down_proj
        w_out_stf = down_proj_stf.weight[:, neuron_idx_stf].detach().float()

        w_folded_stf = final_norm_gamma_stf * w_out_stf
        weight_norm_stf = w_folded_stf.norm().item()

        proj_onto_null_stf = null_space_basis_stf @ w_folded_stf
        null_space_norm_stf = proj_onto_null_stf.norm().item()
        null_space_fraction_stf = null_space_norm_stf / (weight_norm_stf + 1e-10)

        direct_contribution_stf = W_U_stf @ w_folded_stf
        logit_var_stf = direct_contribution_stf.var().item()

        return {
            "weight_norm": weight_norm_stf,
            "null_space_fraction": null_space_fraction_stf,
            "logit_var": logit_var_stf,
        }

    # ------------------------------------------------------------
    # 5. Compute for all 15 candidates
    # ------------------------------------------------------------
    candidate_results_stf = []
    for c_stf in candidates_stf:
        stats_stf = compute_stolfo_criteria_stf(c_stf["layer"], c_stf["neuron_idx"])
        candidate_results_stf.append({"neuron_id": c_stf["neuron_id"], "group": "candidate", **stats_stf})
        print(f"{c_stf['neuron_id']}: weight_norm={stats_stf['weight_norm']:.4f}, "
              f"null_space_fraction={stats_stf['null_space_fraction']:.4f}, "
              f"logit_var={stats_stf['logit_var']:.6f}")

    # ------------------------------------------------------------
    # 6. Null distribution: ~1,000 random same-layer neurons
    # ------------------------------------------------------------
    random_stf.seed(42)
    candidate_layers_stf = {c_stf["layer"] for c_stf in candidates_stf}
    candidate_ids_stf = {(c_stf["layer"], c_stf["neuron_idx"]) for c_stf in candidates_stf}
    intermediate_size_stf = model_stf.config.intermediate_size

    random_neurons_stf = []
    while len(random_neurons_stf) < 1000:
        l_stf = random_stf.choice(list(candidate_layers_stf))
        n_stf = random_stf.randint(0, intermediate_size_stf - 1)
        if (l_stf, n_stf) not in candidate_ids_stf:
            random_neurons_stf.append((l_stf, n_stf))

    random_results_stf = []
    for i_stf, (layer_idx_stf, neuron_idx_stf) in enumerate(random_neurons_stf):
        stats_stf = compute_stolfo_criteria_stf(layer_idx_stf, neuron_idx_stf)
        random_results_stf.append({
            "neuron_id": f"L{layer_idx_stf}_N{neuron_idx_stf}", "group": "random", **stats_stf
        })
        if i_stf % 200 == 0:
            print(f"  random neuron {i_stf}/1000...")

    # ------------------------------------------------------------
    # 7. Save and summarize
    # ------------------------------------------------------------
    all_results_stf = pd_stf.DataFrame(candidate_results_stf + random_results_stf)
    all_results_stf.to_csv("stolfo_criteria.csv", index=False)  # repo-level, shared output

    cand_df_stf = all_results_stf[all_results_stf["group"] == "candidate"]
    rand_df_stf = all_results_stf[all_results_stf["group"] == "random"]

    print("\n--- SUMMARY ---")
    print(f"Candidates: mean null_space_fraction={cand_df_stf['null_space_fraction'].mean():.4f}, "
          f"mean logit_var={cand_df_stf['logit_var'].mean():.6f}")
    print(f"Random:     mean null_space_fraction={rand_df_stf['null_space_fraction'].mean():.4f}, "
          f"mean logit_var={rand_df_stf['logit_var'].mean():.6f}")

    weight_norm_90th_stf = rand_df_stf["weight_norm"].quantile(0.90)
    null_frac_90th_stf = rand_df_stf["null_space_fraction"].quantile(0.90)

    genuine_stf = cand_df_stf[
        (cand_df_stf["weight_norm"] > weight_norm_90th_stf) &
        (cand_df_stf["null_space_fraction"] > null_frac_90th_stf)
    ]
    print(f"\n{len(genuine_stf)} / {len(cand_df_stf)} candidates exceed random-neuron 90th percentile "
          f"on BOTH weight_norm and null_space_fraction (genuine Stolfo-style entropy neurons):")
    print(genuine_stf["neuron_id"].tolist())

    # Quick check on the two neurons of specific interest from earlier phases
    for check_id in ["L31_N2477", "L30_N3533", "L26_N2788"]:
        row = cand_df_stf[cand_df_stf["neuron_id"] == check_id]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n{check_id}: weight_norm={r['weight_norm']:.4f} "
                  f"(random 90th={weight_norm_90th_stf:.4f}), "
                  f"null_space_fraction={r['null_space_fraction']:.4f} "
                  f"(random 90th={null_frac_90th_stf:.4f})")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PHASE 2, part 4
    """)
    return


@app.cell
def _(baseline_prompts):
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
    with open("data/lack_of_knowledge/prompts.jsonl") as f_fnc:
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
        inputs_fnc = tokenizer_fnc(prompt_text_fnc, return_tensors="pt", add_special_tokens=False).to(model_fnc.device)
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
    df_fnc.to_csv("person_B_lack_of_knowledge/results/frozen_norm_L31_N2477.csv", index=False)

    print(f"Mean shift under frozen norm: {df_fnc['shift_under_frozen_norm'].mean():.6f}")
    return tokenizer_fnc, working_prompts_fnc


@app.cell
def _(tokenizer_fnc, working_prompts_fnc):
    # Confirm both tokenizer calls are actually using the fix, and confirm the
    # two forward passes see the SAME token sequence
    prompt_text_fnc1 = working_prompts_fnc[0]["chat_formatted_prompt"]

    check1 = tokenizer_fnc(prompt_text_fnc1, return_tensors="pt", add_special_tokens=False)
    check2 = tokenizer_fnc(prompt_text_fnc1, return_tensors="pt")  # old, unfixed way

    print("Fixed tokenization length:", check1["input_ids"].shape)
    print("Unfixed tokenization length:", check2["input_ids"].shape)
    print("First 5 tokens, fixed:", check1["input_ids"][0][:5])
    print("First 5 tokens, unfixed:", check2["input_ids"][0][:5])
    return


@app.cell
def _(subprocess):


    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(status.stdout)
    return


@app.cell
def _(subprocess):
    pull = subprocess.run(
        ["git", "pull", "--rebase"],
        capture_output=True,
        text=True,
    )

    print(pull.stdout)
    print(pull.stderr)

    if pull.returncode != 0:
        print("Pull failed — STOP before adding/committing anything.")
    return


@app.cell
def _(subprocess):
    files_to_add = [
        "person_B_lack_of_knowledge/results/frozen_norm_L31_N2477.csv",
        "person_B_lack_of_knowledge/results/results_bf16_unquantized.csv",
        "person_B_lack_of_knowledge/results/working_vs_heldout.csv",
        "stolfo_criteria.csv",
    ]

    subprocess.run(
        ["git", "add", "--"] + files_to_add,
        check=True,
    )
    return (files_to_add,)


@app.cell
def _(subprocess):
    status_after_add = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(status_after_add.stdout)
    return (status_after_add,)


@app.cell
def _(subprocess):
    commit = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Add Phase 2 analysis results for Lack of Knowledge",
        ],
        capture_output=True,
        text=True,
    )

    print(commit.stdout)
    print(commit.stderr)
    return (commit,)


@app.cell
def _(subprocess):
    push = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True,
        text=True,
    )

    print(push.stdout)
    print(push.stderr)
    return


@app.cell
def _():
    #exec(open("person_B_lack_of_knowledge/preprocess_lack_of_knowledge.py").read())
    #records, control_records = main()
    return


@app.cell
def _(Path, os):
    #import os
    #from pathlib import Path

    print("cwd:", os.getcwd())
    p = Path("UnknownBench")
    print("UnknownBench exists:", p.exists())
    if p.exists():
        print("contents:", list(p.iterdir()))
        nec = p / "data" / "NEC"
        print("data/NEC exists:", nec.exists())
        if nec.exists():
            print("data/NEC contents:", list(nec.iterdir()))
    return


@app.cell
def _(shutil):
    #import shutil
    shutil.rmtree("UnknownBench", ignore_errors=True)
    return


@app.cell
def _(Path):

    from person_B_lack_of_knowledge.old_preprocess_lack_of_knowledge import clone_repo

    clone_repo()
    print("exists after clone_repo():", Path("UnknownBench").exists())
    if Path("UnknownBench").exists():
        print(list(Path("UnknownBench").iterdir()))
    return


@app.cell
def _():
    #from shared.prompt_format import verify_induction_quality

    #print("=== unanswerable (expect LOW top1) ===")
    #verify_induction_quality(model, tokenizer, records[:25])

    #print("\n=== answerable controls (expect HIGH top1) ===")
    #verify_induction_quality(model, tokenizer, control_records[:25])
    return


@app.cell
def _(subprocess):
    subprocess.run(["git", "stash", "push", "-u", "-m", "WIP: position-fixed lack_of_knowledge data, pre-template-whitelist"], check=True) 
    return


@app.cell
def _(subprocess):
    subprocess.run(["git", "stash", "pop"], check=True)
    return


@app.cell
def _(model, screen_all_templates, tokenizer):
    exec(open("person_B_lack_of_knowledge/screen_templates.py").read())
    template_results = screen_all_templates(model, tokenizer)
    return (template_results,)


@app.cell
def _(print_whitelist_recommendation, template_results):
    whitelist = print_whitelist_recommendation(template_results, min_gap=0.15, max_unanswerable_mean=0.5)
    return


@app.cell
def _(
    build_completion_prompt,
    compute_top1_prob,
    defaultdict,
    get_next_token_probs,
    load_and_classify,
    model,
    template_results,
    tokenizer,
):
    def rescreen_candidates(model, tokenizer, candidate_templates, n_per_side=15, seed=7):
        import random
        random.seed(seed)
        unans, ans = load_and_classify()
        by_u = defaultdict(list)
        by_a = defaultdict(list)
        for r in unans:
            if r["template"] in candidate_templates:
                by_u[r["template"]].append(r["prompt"])
        for r in ans:
            if r["template"] in candidate_templates:
                by_a[r["template"]].append(r["prompt"])

        out = []
        for t in candidate_templates:
            u_pool, a_pool = by_u.get(t, []), by_a.get(t, [])
            n = min(n_per_side, len(u_pool), len(a_pool))
            u_s = random.sample(u_pool, n)
            a_s = random.sample(a_pool, n)
            u_top1 = [compute_top1_prob(get_next_token_probs(model, tokenizer, build_completion_prompt(tokenizer, p)["chat_formatted_prompt"])) for p in u_s]
            a_top1 = [compute_top1_prob(get_next_token_probs(model, tokenizer, build_completion_prompt(tokenizer, p)["chat_formatted_prompt"])) for p in a_s]
            um, cm = sum(u_top1)/n, sum(a_top1)/n
            out.append({"template": t, "n": n, "unans_mean": um, "ctrl_mean": cm, "gap": cm-um})
            print(f"n={n:2d}  gap={cm-um:+.3f}  (unans={um:.3f}, ctrl={cm:.3f})  {t}")
        return out

    passing_templates = [r["template"] for r in template_results if r["gap"] >= 0.15 and r["unanswerable_mean_top1"] <= 0.5]
    rescreen_results = rescreen_candidates(model, tokenizer, passing_templates, n_per_side=15)
    return


@app.cell
def _():
    #exec(open("person_B_lack_of_knowledge/rebuild_lack_of_knowledge_whitelisted.py").read())
    #records, control_records = rebuild()
    return


@app.cell
def _(control_records, model, records, tokenizer):
    from shared.prompt_format import verify_induction_quality
    r1 = verify_induction_quality(model, tokenizer, records[:25])
    r2 = verify_induction_quality(model, tokenizer, control_records[:25])
    print(r2["mean_top1"] - r1["mean_top1"])   # want this clearly positive, not ~0.03 like before
    return (verify_induction_quality,)


@app.cell
def _(files_to_add, subprocess):
    files_to_add3 = [
        "data/lack_of_knowledge/prompts.jsonl",
        "data/lack_of_knowledge/controls.jsonl",
    ]

    subprocess.run(
        ["git", "add", "--"] + files_to_add,
        check=True,
    )
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "git",
            "add",
            "data/lack_of_knowledge/prompts.jsonl",
            "data/lack_of_knowledge/controls.jsonl",
        ],
        check=True,
    )
    return


@app.cell
def _(status_after_add, subprocess):
    status_after_add3 = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(status_after_add.stdout)
    return


@app.cell
def _(result, subprocess):
    result7 = subprocess.run(
        [
            "git",
            "add",
            "-f",
            "--",
            "data/lack_of_knowledge/prompts.jsonl",
            "data/lack_of_knowledge/controls.jsonl",
        ],
        capture_output=True,
        text=True,
    )

    print("return code:", result.returncode)
    print(result.stdout)
    print(result.stderr)
    return


@app.cell
def _(result, subprocess):
    result8 = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(result.stdout)
    return


@app.cell
def _(commit, subprocess):
    commit3 = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Phase 3 rem for Lack of Knowledge, control prompts",
        ],
        capture_output=True,
        text=True,
    )

    print(commit.stdout)
    print(commit.stderr)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        ["git", "config", "user.name", "Sheryl Mehta"],
        check=True,
    )

    subprocess.run(
        ["git", "config", "user.email", "sherylmehta11@gmail.com"],
        check=True,
    )
    return


@app.cell
def _(commit, subprocess):
    commit6 = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Phase 3 rem for Lack of Knowledge, control prompts",
        ],
        capture_output=True,
        text=True,
    )

    print(commit.stdout)
    print(commit.stderr)
    return


@app.cell
def _(result, subprocess):
    for key in ["user.name", "user.email"]:
        result0 = subprocess.run(
            ["git", "config", "--local", key],
            capture_output=True,
            text=True,
            check=True,
        )
        print(key, "=", result.stdout.strip())
    return


@app.cell
def _(subprocess):


    git_name_set = subprocess.run(
        ["git", "config", "--local", "user.name", "Sheryl Mehta"],
        capture_output=True,
        text=True,
    )

    git_email_set = subprocess.run(
        ["git", "config", "--local", "user.email", "sherylmehta11@gmail.com"],
        capture_output=True,
        text=True,
    )

    print("Name config:", git_name_set.returncode)
    print("Email config:", git_email_set.returncode)
    return


@app.cell
def _(subprocess):
    name_check = subprocess.run(
        ["git", "config", "--local", "--get", "user.name"],
        capture_output=True,
        text=True,
        check=True,
    )

    email_check = subprocess.run(
        ["git", "config", "--local", "--get", "user.email"],
        capture_output=True,
        text=True,
        check=True,
    )

    print("Git name:", name_check.stdout.strip())
    print("Git email:", email_check.stdout.strip())
    return


@app.cell
def _(commit, subprocess):
    commit60 = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Phase 3 rem for Lack of Knowledge, control prompts",
        ],
        capture_output=True,
        text=True,
    )

    print(commit.stdout)
    print(commit.stderr)
    return


@app.cell
def _(subprocess):


    commit_now = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Phase 3 Rem: Add lack of knowledge control prompts",
        ],
        capture_output=True,
        text=True,
    )

    print("Return code:", commit_now.returncode)
    print(commit_now.stdout)
    print(commit_now.stderr)
    return


@app.cell
def _(subprocess):
    log_check = subprocess.run(
        ["git", "log", "-1", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(log_check.stdout)
    return


@app.cell
def _(subprocess):
    push_now = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True,
        text=True,
    )

    print("Return code:", push_now.returncode)
    print(push_now.stdout)
    print(push_now.stderr)
    return


@app.cell
def _(pilot_check_single_answer_schema):
    exec(open("person_A_ambiguity/preprocessing_ambiguity_2/preprocess_ambiguity_v2.py").read())
    pilot_check_single_answer_schema()
    return


@app.cell
def _(control_records, model, records, tokenizer, verify_induction_quality):
    r12 = verify_induction_quality(model, tokenizer, records[:25])
    r21 = verify_induction_quality(model, tokenizer, control_records[:25])
    print(r21["mean_top1"] - r12["mean_top1"])
    return


@app.cell
def _(main):
    records, control_records = main()
    return control_records, records


@app.cell
def _(subprocess):
    git_status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(git_status.stdout)
    return


@app.cell
def _(subprocess):
    git_status1 = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    status_lines = [
        line for line in git_status1.stdout.splitlines()
        if line.strip()
    ]

    print(f"Changed files: {len(status_lines)}")
    print("\n".join(status_lines) if status_lines else "Working tree clean")
    return


@app.cell
def _(subprocess):
    files_to_add7 = [
        "data/ambiguity/prompts.jsonl",
        "data/ambiguity/control_review_log.jsonl",
        "data/ambiguity/controls.jsonl",
    ]

    subprocess.run(
        ["git", "add", "--"] + files_to_add7,
        check=True,
    )

    print("Files staged.")
    return


@app.cell
def _(subprocess):
    git_staged_check = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(git_staged_check.stdout)
    return


@app.cell
def _(subprocess):
    commit_result = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Ambiguity phase 3 control prompts",
        ],
        capture_output=True,
        text=True,
    )

    print("Return code:", commit_result.returncode)
    print(commit_result.stdout)
    print(commit_result.stderr)
    return (commit_result,)


@app.cell
def _(subprocess):
    push_result = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True,
        text=True,
    )

    print("Return code:", push_result.returncode)
    print(push_result.stdout)
    print(push_result.stderr)
    return


@app.cell(hide_code=True)
def _(subprocess):
    final_git_status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(final_git_status.stdout)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PHASE 3 CANDIDATE NEURONS
    """)
    return


@app.cell
def _(json, model, tokenizer):
    baseline_prompts = []
    for cat_file in ["data/ambiguity/prompts.jsonl", "data/lack_of_knowledge/prompts.jsonl",
                      "data/contradictory_context/prompts.jsonl"]:
        with open(cat_file) as ff:
            baseline_prompts += [json.loads(l)["chat_formatted_prompt"] for l in ff
                                  if json.loads(l)["split"] == "working"]

    print(f"Baseline pool: {len(baseline_prompts)} prompts")  # expect 252

    from shared.detection import detect_candidate_neurons_split_half, save_candidate_neurons_v2
    candidates, full_dist = detect_candidate_neurons_split_half(
        model, tokenizer, baseline_prompts, layer_range=range(20, 32),
    )
    save_candidate_neurons_v2(candidates, full_dist, baseline_prompts, seed=42)
    return baseline_prompts, candidates


@app.cell
def _():
    import pathlib
    pathlib.Path("shared/rescue_untruncated_candidates.py").write_text(r'''"""
    rescue_untruncated_candidates.py -- recovers the full split-half-stable
    neuron set WITHOUT re-running detection.

    detect_candidate_neurons_split_half() already found 17 neurons that landed
    in the top-60 (by |correlation|) of BOTH independently-computed halves --
    that agreement IS the validation criterion. save_candidate_neurons_v2()
    then truncated to top_k_final=15 before writing candidate_neurons.json,
    silently dropping 2 neurons that had already passed the real bar. 15 was
    just matching the old Phase 2 candidate count for comparison, not a
    methodologically meaningful cutoff -- once split-half agreement is met,
    there's no principled reason to drop the last 2 by rank alone.

    full_correlation_distribution.json already contains both halves' complete
    correlation dictionaries, so this just re-derives top_a/top_b/stable from
    that file (same top_k_per_half=60 used originally) and re-saves
    candidate_neurons.json with the FULL stable set, untruncated. No model
    needed, no GPU needed -- this is pure post-processing on what you already
    computed.

    Run with:
        exec(open("shared/rescue_untruncated_candidates.py").read())
        rescue()
    """

    import json


    def rescue(
        distribution_path="full_correlation_distribution.json",
        candidates_path="candidate_neurons.json",
        top_k_per_half=60,
    ):
        with open(distribution_path) as f:
            dist = json.load(f)

        def parse_key(k):
            # "L{layer}_N{neuron}" -> (layer, neuron)
            l_part, n_part = k.split("_")
            return int(l_part[1:]), int(n_part[1:])

        corr_a = {parse_key(k): v for k, v in dist["half_a"].items()}
        corr_b = {parse_key(k): v for k, v in dist["half_b"].items()}

        top_a = set(sorted(corr_a, key=lambda k: -abs(corr_a[k]))[:top_k_per_half])
        top_b = set(sorted(corr_b, key=lambda k: -abs(corr_b[k]))[:top_k_per_half])
        stable = top_a & top_b
        print(f"{len(top_a)} in top-{top_k_per_half} of half A, {len(top_b)} of half B, "
              f"{len(stable)} in BOTH (split-half-stable) -- keeping ALL of these, no truncation.")

        ranked = sorted(stable, key=lambda k: -min(abs(corr_a[k]), abs(corr_b[k])))

        candidates = [
            {
                "neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n,
                "detection_correlation_half_a": corr_a[(l, n)],
                "detection_correlation_half_b": corr_b[(l, n)],
                "detection_correlation_min_abs": min(abs(corr_a[(l, n)]), abs(corr_b[(l, n)])),
            }
            for (l, n) in ranked
        ]

        with open(candidates_path) as f:
            existing = json.load(f)
        provenance = existing.get("provenance", {})
        provenance["method"] = "split_half_validated"
        provenance["note"] = (
            f"Corrected via rescue script: original save truncated "
            f"to top_k_final=15, dropping {len(candidates) - 15 if len(candidates) >= 15 else 0} "
            f"neuron(s) that had already passed split-half validation. This file keeps the "
            f"full validated set ({len(candidates)} neurons)."
        )

        payload = {"provenance": provenance, "candidates": candidates}
        with open(candidates_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Re-saved {len(candidates)} split-half-validated candidates to {candidates_path} "
              f"(was 15, truncated).")
        for c in candidates:
            print(f"  {c['neuron_id']}: min|r|={c['detection_correlation_min_abs']:.4f}")

        return candidates


    if __name__ == "__main__":
        rescue()
    ''')
    print("written")
    return


@app.cell
def _(rescue):
    exec(open("shared/rescue_untruncated_candidates.py").read())
    candidatess = rescue()
    return


@app.cell
def _(subprocess):
    git_statuss = subprocess.run(
        ["git", "status", "--short", "--branch"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(git_statuss.stdout)
    return


@app.cell
def _(subprocess):
    files_to_add9 = [
        "candidate_neurons.json",
        "full_correlation_distribution.json",
        "shared/rescue_untruncated_candidates.py",
    ]

    subprocess.run(
        ["git", "add", "--"] + files_to_add9,
        check=True,
    )

    print("Files staged.")
    return


@app.cell
def _(subprocess):
    git_staged_checkk = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(git_staged_checkk.stdout)
    return


@app.cell
def _(commit_result, subprocess):
    commit_resultt = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Phase 3 Mechanism check + ablation",
        ],
        capture_output=True,
        text=True,
    )

    print("Return code:", commit_result.returncode)
    print(commit_resultt.stdout)
    print(commit_resultt.stderr)
    return


@app.cell
def _(subprocess):
    push_resultt = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True,
        text=True,
    )

    print("Return code:", push_resultt.returncode)
    print(push_resultt.stdout)
    print(push_resultt.stderr)
    return


@app.cell
def _(subprocess):
    final_git_statuss = subprocess.run(
        ["git", "status", "--short", "--branch"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(final_git_statuss.stdout)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PHASE 3 NEXT STEP 2
    """)
    return


@app.cell
def _(model, tokenizer):
    from shared.run_phase34 import run_category

    run_category(
        model, tokenizer, category="lack_of_knowledge",
        prompts_path="data/lack_of_knowledge/prompts.jsonl",
        controls_path="data/lack_of_knowledge/controls.jsonl",
        candidates_path="candidate_neurons.json",
        out_results_path="person_B_lack_of_knowledge/results/results_v3.csv",
        out_control_results_path="person_B_lack_of_knowledge/results/control_results_v3.csv",
    )

    run_category(
        model, tokenizer, category="ambiguity",
        prompts_path="data/ambiguity/prompts.jsonl",
        controls_path="data/ambiguity/controls.jsonl",
        candidates_path="candidate_neurons.json",
        out_results_path="person_A_ambiguity/results/results_v3.csv",
        out_control_results_path="person_A_ambiguity/results/control_results_v3.csv",
    )

    run_category(
        model, tokenizer, category="contradictory_context",
        prompts_path="data/contradictory_context/prompts.jsonl",
        controls_path="data/contradictory_context/controls.jsonl",
        candidates_path="candidate_neurons.json",
        out_results_path="person_C_contradictory_context/results/results_v3.csv",
        out_control_results_path="person_C_contradictory_context/results/control_results_v3.csv",
    )
    return


@app.cell
def _(subprocess):
    subprocess.run(
        ["git", "add", "-A"],
        check=True,
    )

    print("All changes staged.")
    return


if __name__ == "__main__":
    app.run()
