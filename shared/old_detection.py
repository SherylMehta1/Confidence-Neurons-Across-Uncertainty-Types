"""Deprecated shim: everything moved to shared.detection."""
import warnings
warnings.warn("shared.old_detection is deprecated; import from shared.detection", DeprecationWarning, stacklevel=2)
from shared.detection import *  # noqa: F401,F403,E402
from shared.detection import capture_intermediate_activations, get_neuron_activation, load_candidate_neurons  # noqa: F401,E402
