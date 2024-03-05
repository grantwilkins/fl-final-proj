"""Module containing derivatives of Permute."""

from torch import Tensor, argsort

from backpack_local.core.derivatives.basederivatives import BaseDerivatives
from backpack_local.custom_module.permute import Permute


class PermuteDerivatives(BaseDerivatives):
    """Derivatives of Permute."""

    def _jac_t_mat_prod(
        self,
        module: Permute,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        mat: Tensor,
        subsampling: list[int] = None,
    ) -> Tensor:
        return mat.permute(
            [0] + [element + 1 for element in argsort(Tensor(module.dims))]
        )

    def _jac_mat_prod(
        self, module: Permute, g_inp: tuple[Tensor], g_out: tuple[Tensor], mat: Tensor
    ) -> Tensor:
        return mat.permute([0] + [element + 1 for element in module.dims])
