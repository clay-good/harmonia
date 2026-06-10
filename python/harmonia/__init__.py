"""Harmonia — curated, variability-aware cardiac ion-channel drug-block data and
the in-silico ventricular AP models that turn it into a torsade-de-pointes
(proarrhythmia) risk *distribution*.

NOT a clinical tool and NOT a regulatory safety determination. Harmonia reports
a risk-metric distribution and a classification-flip frequency with full input
uncertainty; it never issues a bare "safe/unsafe" verdict. See spec.md §10.
"""
from __future__ import annotations

__version__ = "0.5.0"

from .load import Dataset, load, find_dataset_dir
from .validate import validate_dataset, ValidationReport
from . import filter, records

# The headline API. Imported lazily-safe: simulate pulls in numpy/scipy, which
# are hard dependencies, so a plain ``import harmonia`` already needs them.
from .simulate import (assess, assess_combination, flip_view, flip_sensitivity,
                       RiskAssessment, CombinationAssessment, FlipView,
                       FlipSensitivity, ChannelSensitivity, SobolSensitivity,
                       SobolChannel)
from .populations import (assess_population, PopulationAssessment,
                          calibrate_population, CalibrationResult)
from .exposure import free_from_total, total_from_free
from .infer import (posterior, infer_channel, Posterior, Prior, resolve_prior,
                    learn_tau_pop, fit_dose_response, simulation_based_calibration,
                    posterior_coverage)

CLINICAL_USE = (
    "PROHIBITED — research / safety-methodology / education only; "
    "not a regulatory determination"
)

__all__ = [
    "__version__",
    "Dataset",
    "load",
    "find_dataset_dir",
    "validate_dataset",
    "ValidationReport",
    "filter",
    "records",
    "assess",
    "assess_combination",
    "assess_population",
    "calibrate_population",
    "CalibrationResult",
    "flip_view",
    "flip_sensitivity",
    "RiskAssessment",
    "CombinationAssessment",
    "PopulationAssessment",
    "FlipView",
    "FlipSensitivity",
    "ChannelSensitivity",
    "SobolSensitivity",
    "SobolChannel",
    "free_from_total",
    "total_from_free",
    "posterior",
    "infer_channel",
    "Posterior",
    "Prior",
    "resolve_prior",
    "learn_tau_pop",
    "fit_dose_response",
    "simulation_based_calibration",
    "posterior_coverage",
    "CLINICAL_USE",
]
