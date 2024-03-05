"""Contains derivatives for SumModule."""

from torch import Tensor

from backpack_local.core.derivatives.basederivatives import BaseDerivatives
from backpack_local.custom_module.branching import SumModule


class SumModuleDerivatives(BaseDerivatives):
    """Contains derivatives for SumModule."""

    def _jac_t_mat_prod(
        self,
        module: SumModule,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        mat: Tensor,
        subsampling: list[int] = None,
    ) -> Tensor:
        return mat

    def _jac_mat_prod(
        self,
        module: SumModule,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        mat: Tensor,
        subsampling: list[int] = None,
    ) -> Tensor:
        return mat
