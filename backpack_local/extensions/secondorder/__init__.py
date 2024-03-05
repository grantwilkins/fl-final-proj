"""Second order extensions
====================================

Second-order extensions propagate additional information through the graph
to extract structural or local approximations to second-order information.
They are more expensive to run than a standard gradient backpropagation.
The implemented extensions are

- The diagonal of the Generalized Gauss-Newton (GGN)/Fisher information,
  using exact computation
  (:func:`DiagGGNExact <backpack.extensions.DiagGGNExact>`)
  or Monte-Carlo approximation
  (:func:`DiagGGNMC <backpack.extensions.DiagGGNMC>`).
- Kronecker Block-Diagonal approximations of the GGN/Fisher
  :func:`KFAC <backpack.extensions.KFAC>`,
  :func:`KFRA <backpack.extensions.KFRA>`,
  :func:`KFLR <backpack.extensions.KFLR>`.
- The diagonal of the Hessian :func:`DiagHessian <backpack.extensions.DiagHessian>`
- The symmetric (square root) factorization of the GGN/Fisher information,
  using exact computation
  (:func:`SqrtGGNExact <backpack.extensions.SqrtGGNExact>`)
  or a Monte-Carlo (MC) approximation
  (:func:`SqrtGGNMC<backpack.extensions.SqrtGGNMC>`)
"""

from backpack_local.extensions.secondorder.diag_ggn import (
    BatchDiagGGNExact,
    BatchDiagGGNMC,
    DiagGGNExact,
    DiagGGNMC,
)
from backpack_local.extensions.secondorder.diag_hessian import (
    BatchDiagHessian,
    DiagHessian,
)
from backpack_local.extensions.secondorder.hbp import HBP, KFAC, KFLR, KFRA
from backpack_local.extensions.secondorder.sqrt_ggn import SqrtGGNExact, SqrtGGNMC

__all__ = [
    "HBP",
    "KFAC",
    "KFLR",
    "KFRA",
    "BatchDiagGGNExact",
    "BatchDiagGGNMC",
    "BatchDiagHessian",
    "DiagGGNExact",
    "DiagGGNMC",
    "DiagHessian",
    "SqrtGGNExact",
    "SqrtGGNMC",
]
