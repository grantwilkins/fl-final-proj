"""BackPACK extensions that can be passed into a ``with backpack(...)`` context."""

from .curvmatprod import GGNMP, HMP, PCHMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance
from .secondorder import (
    HBP,
    KFAC,
    KFLR,
    KFRA,
    BatchDiagGGNExact,
    BatchDiagGGNMC,
    BatchDiagHessian,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SqrtGGNExact,
    SqrtGGNMC,
)

__all__ = [
    "GGNMP",
    "HBP",
    "HMP",
    "KFAC",
    "KFLR",
    "KFRA",
    "PCHMP",
    "BatchDiagGGNExact",
    "BatchDiagGGNMC",
    "BatchDiagHessian",
    "BatchGrad",
    "BatchL2Grad",
    "DiagGGNExact",
    "DiagGGNMC",
    "DiagHessian",
    "SqrtGGNExact",
    "SqrtGGNMC",
    "SumGradSquared",
    "Variance",
]
