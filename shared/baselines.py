"""
shared/baselines.py -- the fixed, documented general-purpose baseline prompt
set used to estimate each neuron's mean activation for mean-ablation
(mean_source="general_baseline").

The original ablation run's baseline set was never committed; this list (60
generic, domain-varied sentence openers, ported from
results/rerun_full_precision.py in the independent re-run) is the reproducible
replacement. Means estimated on it will differ slightly from the original run.

Two ways to present these to the model:
  - "raw":  the bare sentence, tokenized with a single BOS (tokenize_prompt).
  - "chat": wrapped by format_chat_prompt as the user turn (assistant-turn
            opener position) -- closer to the chat-formatted data distribution
            but the measured position is a turn opener, not a continuation.
Whichever is used is recorded in provenance (baseline_format +
baseline_prompt_sha256 of the exact strings fed to the model).
"""

from shared.provenance import sha256_prompts

GENERAL_BASELINE_PROMPTS = [
    "The weather today is", "My favorite hobby is", "The history of Rome began",
    "Water boils at a temperature of", "The largest planet in our solar system is",
    "Yesterday I went to the store and bought", "The capital of Japan is",
    "Photosynthesis is the process by which", "The novel was written in a style that",
    "To bake bread you first need to", "The stock market closed today with",
    "Einstein is best known for his theory of", "The recipe calls for two cups of",
    "In the nineteenth century, most people", "The committee decided to postpone the",
    "A triangle has three sides and", "The concert was cancelled because of",
    "Most species of birds are able to", "The programming language Python was created by",
    "She opened the letter and found", "The chemical symbol for gold is",
    "During the winter months, bears typically", "The museum's new exhibit features",
    "According to the weather forecast, tomorrow will be", "The first person to walk on the moon was",
    "My grandmother always used to say", "The engine failed because the fuel",
    "In mathematics, a prime number is", "The restaurant on the corner serves",
    "The ocean covers about seventy percent of", "He picked up the guitar and started to",
    "The French Revolution began in the year", "A healthy diet should include plenty of",
    "The train departs from platform nine at", "Shakespeare wrote many plays including",
    "The human heart pumps blood through", "After the meeting, the manager sent",
    "The password must contain at least eight", "Lightning is caused by the buildup of",
    "The children spent the afternoon playing", "The ancient pyramids of Egypt were built",
    "To solve this equation, first isolate the", "The airline announced that all flights",
    "Coffee is one of the most widely consumed", "The garden was full of roses and",
    "The judge ruled that the evidence was", "A computer's central processing unit is",
    "The marathon route passes through the", "In chess, the queen can move",
    "The documentary explores the lives of", "The bridge was closed for repairs after",
    "Salt dissolves in water because", "The orchestra tuned their instruments before",
    "The invention of the printing press allowed", "Her research focuses on the effects of",
    "The hotel room overlooked a beautiful", "Gravity causes objects to fall at",
    "The election results were announced on", "The library extended its opening hours to",
    "The patient was advised to rest and",
]
assert len(GENERAL_BASELINE_PROMPTS) == 60

# sha256 of the raw list (order-independent; same recipe as detection's hash).
GENERAL_BASELINE_SHA256 = sha256_prompts(GENERAL_BASELINE_PROMPTS)


def general_baseline_prompts(tokenizer=None, fmt="raw"):
    """Return the baseline strings as fed to the model. fmt="raw" returns the
    bare sentences; fmt="chat" wraps each as a user turn via format_chat_prompt
    (requires a tokenizer with a chat template)."""
    if fmt == "raw":
        return list(GENERAL_BASELINE_PROMPTS)
    if fmt == "chat":
        from shared.prompt_format import format_chat_prompt
        return [format_chat_prompt(tokenizer, p) for p in GENERAL_BASELINE_PROMPTS]
    raise ValueError(f"unknown baseline format {fmt!r} (raw|chat)")


def load_baseline_file(path):
    """A user-supplied baseline file: .jsonl records with chat_formatted_prompt
    (or 'prompt'/'text'), or a plain text file with one prompt per line."""
    import json
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if str(path).endswith(".jsonl"):
                rec = json.loads(line)
                prompts.append(rec.get("chat_formatted_prompt") or rec.get("prompt") or rec["text"])
            else:
                prompts.append(line)
    return prompts
