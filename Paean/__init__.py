"""Independent Paean final-evaluation package.

The package keeps final-eval command-line configuration separate from training
configuration while reusing the existing UnifiedFinalEvaluationModule engine.
"""

from .config import FinalEvalSettings, parse_final_eval_settings
from .embedded import run_embedded_final_eval

__all__ = [
    "FinalEvalSettings",
    "parse_final_eval_settings",
    "run_embedded_final_eval",
]
