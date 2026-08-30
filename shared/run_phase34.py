"""Deprecated shim: the per-category ablation runner is now
shared.run_ablation_pipeline (run_category signature changed -- see its docstring)."""
import warnings
warnings.warn("shared.run_phase34 is deprecated; use shared.run_ablation_pipeline", DeprecationWarning, stacklevel=2)
from shared.run_ablation_pipeline import run_category, run_category_from_paths, load_candidate_neurons  # noqa: F401,E402
