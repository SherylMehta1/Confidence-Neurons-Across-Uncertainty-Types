# (paste full diagnose_templates.py content)
"""
diagnose_templates.py -- one-time diagnostic for Person C's category.
Scans CounterFact for relations whose prompt template looks like a
grammatical fragment, then checks whether ParaRel has a usable
replacement template for the same Wikidata relation_id.

Run this BEFORE editing preprocess_contradictory_context.py.
"""

import re
from datasets import load_dataset

# --- Step A: pull every unique relation_id -> template from CounterFact ---
ds = load_dataset("azhx/counterfact", split="train")


def get_field(record, *candidates):
    for path in candidates:
        obj = record
        try:
            for key in path:
                obj = obj[key]
            if obj not in (None, ""):
                return obj
        except (KeyError, TypeError, IndexError):
            continue
    return None


seen = {}
for record in ds:
    template = get_field(record, ("requested_rewrite", "prompt"), ("prompt",))
    relation_id = get_field(record, ("requested_rewrite", "relation_id"), ("relation_id",))
    if template is None or relation_id is None:
        continue
    if relation_id not in seen:
        seen[relation_id] = template

print(f"{len(seen)} unique relations in CounterFact.")

# --- Step B: heuristic flag for fragment-looking templates ---
suspicious = {}
for rel_id, tmpl in seen.items():
    tail = tmpl.split("{}")[-1].strip()
    if tail == "" or not re.search(r"\b(is|are|was|were|speaks|of|by|in|for)\b", tail, re.I):
        suspicious[rel_id] = tmpl

print(f"{len(suspicious)} flagged as suspicious (heuristic — eyeball these manually too):\n")
for rel_id, tmpl in suspicious.items():
    print(f"  {rel_id}: {tmpl!r}")

# --- Step C: check ParaRel for a replacement template on flagged relations ---
print("\nLoading coastalcph/pararel_patterns to check for replacements...")
pararel = load_dataset("coastalcph/pararel_patterns", split="train")
print("ParaRel columns:", pararel.column_names)
print("ParaRel first row:", pararel[0])
print("\n^^ CONFIRM the actual field names above before trusting the lookup below. "
      "Adjust 'relation_id'/'pattern' keys in Script 2 if they differ.\n")

pararel_lookup = {}
for row in pararel:
    rel_id = row.get("relation")
    if rel_id and rel_id.endswith(".jsonl"):
        rel_id = rel_id[:-len(".jsonl")]
    raw_template = row.get("template")
    if rel_id and raw_template and rel_id not in pararel_lookup:
        # convert ParaRel's [X]/[Y] style to CounterFact's {} style,
        # keep only the [X] slot since Redefine only fills in the subject
        converted = raw_template.replace("[X]", "{}").replace(" [Y]", "").replace("[Y]", "")
        pararel_lookup[rel_id] = converted.strip()

resolved = {r: pararel_lookup[r] for r in suspicious if r in pararel_lookup}
unresolved = [r for r in suspicious if r not in pararel_lookup]

print(f"Auto-resolvable from ParaRel: {len(resolved)}")
for rel_id, tmpl in resolved.items():
    print(f"  {rel_id}: CounterFact={seen[rel_id]!r}  ->  ParaRel={tmpl!r}")

print(f"\nUnresolved (no ParaRel match, will be dropped): {len(unresolved)}")
print(unresolved)
