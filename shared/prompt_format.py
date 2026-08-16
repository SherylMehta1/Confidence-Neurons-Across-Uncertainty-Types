"""
shared/prompt_format.py -- Phase 3 fix: "Fix lack-of-knowledge position"
(applies equally to ambiguity -- same bug, same fix).

THE BUG (audit finding, both original and re-confirmed in the Phase-2 repo
audit): the cloze suffix ("The answer is") was appended INSIDE the user
message, before the chat template inserts the assistant header:

    user: "What is the habitat of Laursapi? The answer is"
    assistant: <-- generation starts HERE, fresh turn

The model's first generated token is a fresh assistant-turn opener ("I",
"Based", "The", ...), not a continuation of "The answer is". Entropy is
measured on THAT token, not on the model's actual uncertainty about the
answer.

THE FIX: append the cloze suffix AFTER the chat template's generation
prompt, so it's a genuine prefix of the assistant's turn:

    ...<|start_header_id|>assistant<|end_header_id|>\n\nThe answer is
                                                        ^-- generation continues from here

Now the very next generated token is a true continuation of "The answer
is ___", which is what the entropy measurement is supposed to capture.

Used by person_A_ambiguity and person_B_lack_of_knowledge. Contradictory
context does NOT use this -- its "Redefine: ... {base_prompt}" construction
already ends the USER turn on the bare relation phrase (e.g. "...is"), and
the model's assistant-turn completion IS the direct continuation of that
phrase already (no extra fresh-turn-opener token in between), so it doesn't
have this bug. Verify this assumption before assuming it's true forever --
see `verify_induction_quality` below, run it on all three categories.
"""

from typing import Callable, Dict, List, Optional


def build_completion_prompt(
    tokenizer,
    question: str,
    cloze_suffix: str = "The answer is",
    ensure_question_mark: bool = True,
) -> Dict[str, str]:
    """
    Returns {"raw_prompt": <bare question>, "chat_formatted_prompt": <chat
    template + genuine assistant-turn prefill>}.

    raw_prompt is kept as the BARE question (no suffix baked in) so it's
    still human-readable in isolation and diffable against the old data --
    the suffix now lives only in chat_formatted_prompt, where it actually
    changes the induction.
    """
    q = question.strip()
    if ensure_question_mark and not q.endswith("?"):
        q += "?"

    chat_prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # chat_prefix already ends in "...<|start_header_id|>assistant<|end_header_id|>\n\n"
    # -- appending here is a true prefill, not a new turn.
    chat_formatted = chat_prefix + cloze_suffix.strip()

    return {"raw_prompt": q, "chat_formatted_prompt": chat_formatted}


def build_records_with_formatter(
    raw_prompts: List[str],
    category: str,
    source_dataset: str,
    prefix: str,
    tokenizer,
    formatter: Callable = build_completion_prompt,
    formatter_kwargs: Optional[dict] = None,
    n_working: int = 120,
    split_ratio: float = 0.7,
    seed: int = 42,
    is_control: bool = False,
) -> List[Dict]:
    """
    Drop-in replacement for shared.schema_utils.build_records that lets you
    pass a custom prompt formatter (e.g. build_completion_prompt) instead of
    the old raw format_chat_prompt. Keeps the exact same output schema
    (RESULTS_SCHEMA.md / DATA_SOURCES.md), plus an `is_control` field so
    Phase 3's matched-control prompts can share the same file format and be
    told apart from the working/held-out uncertain set.
    """
    import random

    random.seed(seed)
    formatter_kwargs = formatter_kwargs or {}
    pool = list(raw_prompts)
    random.shuffle(pool)

    if len(pool) < n_working:
        raise ValueError(
            f"[{category}] Only {len(pool)} candidate prompts available, "
            f"need at least {n_working}. Loosen your filtering criteria."
        )

    subsample = pool[:n_working]
    n_working_split = int(len(subsample) * split_ratio)

    records = []
    for i, raw_prompt in enumerate(subsample):
        formatted = formatter(tokenizer, raw_prompt, **formatter_kwargs)
        records.append({
            "prompt_id": f"{prefix}_{i:04d}",
            "category": category,
            "raw_prompt": formatted["raw_prompt"],
            "chat_formatted_prompt": formatted["chat_formatted_prompt"],
            "source_dataset": source_dataset,
            "split": "working" if i < n_working_split else "held_out",
            "is_control": is_control,
        })
    return records


def verify_induction_quality(model, tokenizer, prompts: List[dict], max_top1: float = 0.95) -> dict:
    """
    Phase 3 "done when" check for the position fix: verify top-1 probability
    drops well below 0.95 on the FIXED prompts. Run this on a sample (~20-30
    prompts is enough to sanity check) before regenerating the full dataset.

    prompts: list of record dicts with "chat_formatted_prompt" already set
    (i.e. the output of build_completion_prompt / build_records_with_formatter).

    Returns a summary dict; prints per-prompt top1 so you can eyeball which
    ones are still stuck near-1.0 (usually a sign the question isn't
    actually short-answer/factoid shaped and should be filtered out upstream,
    not papered over here).
    """
    from shared.model_utils import get_next_token_probs, compute_top1_prob

    top1s = []
    for r in prompts:
        probs = get_next_token_probs(model, tokenizer, r["chat_formatted_prompt"])
        t1 = compute_top1_prob(probs)
        top1s.append(t1)
        flag = " <-- still peaked" if t1 > max_top1 else ""
        print(f"  [{r['prompt_id']}] top1={t1:.4f}{flag}  {r['raw_prompt'][:70]}")

    import numpy as np
    top1s = np.array(top1s)
    frac_over = float((top1s > max_top1).mean())
    summary = {
        "n": len(top1s),
        "mean_top1": float(top1s.mean()),
        "frac_over_threshold": frac_over,
        "threshold": max_top1,
    }
    print(f"\nmean top1={summary['mean_top1']:.4f}, "
          f"{frac_over*100:.1f}% of prompts still above {max_top1} "
          f"({'PASS' if frac_over < 0.15 else 'INVESTIGATE -- still too peaked'})")
    return summary
