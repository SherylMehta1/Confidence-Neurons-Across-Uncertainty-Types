# che2026steering: Does the Model Use the Feature? Separating Steering from Mechanism in LLMs
Status: verified      Checked: 2026-10-10      URL: https://arxiv.org/abs/2610.07270
Authors / year / venue: Che, Li / 2026 (submitted 5 Oct) / arXiv preprint
WHY it was written: Steering a feature can push it outside its natural range, so steering success need not show the model uses the feature.
HOW: Installation and removal of natural feature values between matched prompts, plus downstream rescue; Gemma 2 2B-IT and Llama 3.1 8B-Instruct entity features; Pythia-70M agreement features.
WHAT it found: Unknown-entity latent steering raises abstention 33.0% -> 79.0%, but installing natural values transfers only 0.12 of the known/unknown contrast; dense directions show opposite installation/removal asymmetries in Gemma vs Llama.
Relation to us: reframes
  Which claim id it touches: C4, C5
  Same entity-familiarity setting; no popularity x knowledge split, no entropy. Our no-hedge control is the analogous caveat.
Anchor: "Tracking a concept and steering a behavior therefore do not by themselves show that the model uses a feature" (abstract)
