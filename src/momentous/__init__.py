"""Covariance-aware angular-moment analysis through one prepared workflow.

Examples
--------
>>> import momentous as mo
>>> pool = mo.Waveset.from_max_l(1)
>>> data = mo.MomentData({(0, 0): 1.0, (1, 0): 0.0}, covariance=mo.Covariance.exact())
>>> analysis = mo.analyze(data, pool)
>>> analysis.search().complete
True
"""

from momentous.analysis import AnalysisResult, CandidateAnalysis, analyze
from momentous.basis import MomentBasis
from momentous.covariance import Covariance
from momentous.data import MomentData
from momentous.domain import Moment, Observable, Wave, Waveset
from momentous.extraction import (
    Acceptance,
    ExtractionError,
    ExtractionResult,
    MCIntegration,
    MCStatistics,
    ResponseDiagnostics,
    ResponseModel,
)
from momentous.results import (
    Bound,
    CheckProgress,
    CheckResult,
    NormalizationWarning,
    ProjectionFailure,
    Region2D,
    Status,
    UnresolvedCompatibilityError,
    UnresolvedPair,
)
from momentous.samples import EventGrouping, EventSample, Polarization
from momentous.search import SearchResult

__all__ = [
    "Acceptance",
    "ResponseModel",
    "AnalysisResult",
    "Bound",
    "CheckProgress",
    "CheckResult",
    "Covariance",
    "EventGrouping",
    "EventSample",
    "ExtractionError",
    "ExtractionResult",
    "MCIntegration",
    "MCStatistics",
    "MomentBasis",
    "Polarization",
    "ResponseDiagnostics",
    "CandidateAnalysis",
    "MomentData",
    "Observable",
    "ProjectionFailure",
    "SearchResult",
    "UnresolvedPair",
    "Moment",
    "NormalizationWarning",
    "Region2D",
    "Status",
    "UnresolvedCompatibilityError",
    "Wave",
    "Waveset",
    "analyze",
]
