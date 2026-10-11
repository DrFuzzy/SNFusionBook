"""snfusion -- reference implementations for *Towards a Unified View of Sensor Fusion:
Theory, Methods, and Autonomous Driving Applications* (Deliparaschos, Oliva,
Charalambous; Springer Nature, 2026).

The package is deliberately small and dependency-light (numpy + scipy only).
Each module maps onto a section of the book:

==================  ==============================================================
module              book section
==================  ==============================================================
``kalman``          Chapter 3, "Bayesian Filtering" (KF, EKF, UKF, NIS)
``models``          Chapter 3, Example ex:ekf-radar-cutin (CV/CA/CT motion models)
``zonotope``        Chapter 3, "Set-Membership and Zonotopic Estimation"
``consensus``       Chapter 3, "Track-Level Fusion" (information fusion, CI) and
                    the consensus iterations used in the distributed chapters
``evidence``        Chapter 3, "Evidence Theory for Conflicting Sensor Information"
``ahp``             Chapter 3, "Fusion based on Multi-Criteria Decision-Making"
``metrics``         Chapter 3, filter-consistency diagnostics (NIS, NEES, RMSE)
``scenarios``       the radar cut-in scenario shared by Examples
                    ex:ekf-radar-cutin, ex:ukf-radar-cutin and ex:zono-radar-cutin
==================  ==============================================================
"""

from . import ahp, consensus, evidence, kalman, metrics, models, scenarios, zonotope
from .kalman import ExtendedKalmanFilter, KalmanFilter, UnscentedKalmanFilter, wrap_angle
from .zonotope import Zonotope
from .evidence import MassFunction

__version__ = "0.1.0"

__all__ = [
    "ahp", "consensus", "evidence", "kalman", "metrics", "models", "scenarios", "zonotope",
    "ExtendedKalmanFilter", "KalmanFilter", "UnscentedKalmanFilter", "wrap_angle",
    "Zonotope", "MassFunction", "__version__",
]
