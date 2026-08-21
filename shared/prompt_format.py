"""
shared/prompt_format.py -- chat-template formatting and prefilled-completion
prompt construction, shared by all three categories.

THE POSITION BUG (audit finding): originally the cloze suffix ("The answer is")
was appended INSIDE the user message, before the chat template inserted the
assistant header:

    user: "What is the habitat of Laursapi? The answer is"
    assistant: <-- generation starts HERE, fresh turn

so the first generated token was a fresh assistant-turn opener ("I", "Based",
"The", ...), not a continuation of "The answer is".

THE FIX: append the suffix AFTER the generation prompt, so it is a genuine
prefill of the assistant turn:

    ...<|start_header_id|>assistant<|end_header_id|>\n\nThe answer is
                                                        ^-- generation continues here

ALL THREE categories use this assistant-turn prefill (see data/*/prompts.jsonl):
ambiguity and lack-of-knowledge prefill "The answer is"; contradictory context
puts "Redefine: <false fact>." in the user turn and prefills the bare relation
phrase ("<subject> is a citizen of") in the assistant turn. An earlier version
of this docstring claimed contradictory context did not need the prefill; that
was wrong -- without it the model would emit a fresh-turn opener there too.

Date pinning: the Llama-3.1 chat template injects "Today Date: <date_string>"
into the system header and defaults date_string to "26 Jul 2024" -- but that
default lives in the template, so every apply_chat_template call here passes
CHAT_TEMPLATE_DATE explicitly, which pins the stored prompt text regardless of
template version. Use format_chat_prompt() rather than calling
apply_chat_template directly.
"""

import random
from typing import Callable, Dict, List, Optional

# The date string baked into every stored chat_formatted_prompt. Changing this
# changes the tokenized prompts and therefore every measurement.
CHAT_TEMPLATE_DATE = "26 Jul 2024"


def format_chat_prompt(tokenizer, user_message: str, date_string: str = CHAT_TEMPLATE_DATE) -> str:
    """Wrap a raw user message in the model's chat template with the generation
    prompt appended (so the next token is the assistant's first token). The
    returned string already starts with the BOS token -- tokenize it with
    shared.model_utils.tokenize_prompt, which will not add a second one."""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, date_string=date_string,
    )


def build_completion_prompt(
    tokenizer,
    question: str,
    cloze_suffix: str = "The answer is",
    ensure_question_mark: bool = True,
) -> Dict[str, str]:
    """
    Returns {"raw_prompt": <bare question>, "chat_formatted_prompt": <chat
    template + genuine assistant-turn prefill>}.

    raw_prompt is kept as the BARE question (no suffix baked in) so it is
    human-readable in isolation and diffable against older data -- the suffix
    lives only in chat_formatted_prompt, where it actually changes the induction.
    """
    q = question.strip()
    if ensure_question_mark and not q.endswith("?"):
        q += "?"

    chat_prefix = format_chat_prompt(tokenizer, q)
    # chat_prefix already ends in "...<|start_header_id|>assistant<|end_header_id|>\n\n"
    # -- appending here is a true prefill, not a new turn.
    chat_formatted = chat_prefix + cloze_suffix.strip()
    return {"raw_prompt": q, "chat_formatted_prompt": chat_formatted}


def seeded_shuffle(items: List, seed: int) -> List:
    """Return a shuffled copy using a private random.Random(seed). For a given
    seed this yields exactly the same order as the legacy `random.seed(seed);
    random.shuffle(...)` (same Mersenne Twister, same shuffle algorithm) -- see
    tests/test_shared_tiny.py -- but without touching global RNG state."""
    pool = list(items)
    random.Random(seed).shuffle(pool)
    return pool


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
    Turn raw prompt strings into the shared data schema (DATA_SOURCES.md):
    seeded subsample of n_working items, 70/30 working/held_out split, with a
    custom formatter (default: build_completion_prompt). `is_control` marks
    matched-control prompts so they can share the file format with the
    uncertain set.
    """
    formatter_kwargs = formatter_kwargs or {}
    pool = seeded_shuffle(raw_prompts, seed)

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
    Check that top-1 probability on the formatted prompts is well below
    max_top1 (i.e. the prefill position actually carries uncertainty). Prints
    per-prompt top1; returns a summary dict.
    """
    from shared.model_utils import get_next_token_probs, compute_top1_prob
    import numpy as np

    top1s = []
    for r in prompts:
        probs = get_next_token_probs(model, tokenizer, r["chat_formatted_prompt"])
        t1 = compute_top1_prob(probs)
        top1s.append(t1)
        flag = " <-- still peaked" if t1 > max_top1 else ""
        print(f"  [{r['prompt_id']}] top1={t1:.4f}{flag}  {r['raw_prompt'][:70]}")

    top1s = np.array(top1s)
    frac_over = float((top1s > max_top1).mean()) if len(top1s) else float("nan")
    summary = {
        "n": int(len(top1s)),
        "mean_top1": float(top1s.mean()) if len(top1s) else float("nan"),
        "frac_over_threshold": frac_over,
        "threshold": max_top1,
    }
    print(f"\nmean top1={summary['mean_top1']:.4f}, "
          f"{frac_over*100:.1f}% of prompts still above {max_top1} "
          f"({'PASS' if frac_over < 0.15 else 'INVESTIGATE -- still too peaked'})")
    return summary
