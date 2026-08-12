import pandas as pd

lack_sig = {"L24_N7891", "L26_N2788", "L29_N11308", "L31_N2477"}
ambig_sig = {"L23_N12156", "L24_N7891", "L29_N10092"}
contra_sig = {"L29_N10092", "L29_N8568"}

all_sets = {
    "lack_of_knowledge": lack_sig,
    "ambiguity": ambig_sig,
    "contradictory_context": contra_sig,
}

# pairwise overlaps
categories = list(all_sets.keys())
for i in range(len(categories)):
    for j in range(i+1, len(categories)):
        a, b = categories[i], categories[j]
        overlap = all_sets[a] & all_sets[b]
        print(f"{a} ∩ {b}: {overlap if overlap else 'none'}")

# three-way overlap
triple_overlap = lack_sig & ambig_sig & contra_sig
print(f"\nAll three categories: {triple_overlap if triple_overlap else 'none'}")

# category-unique neurons
for name, s in all_sets.items():
    others = set().union(*[v for k, v in all_sets.items() if k != name])
    unique = s - others
    print(f"Unique to {name}: {unique}")