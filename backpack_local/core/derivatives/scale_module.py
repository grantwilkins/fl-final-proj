"""Derivatives of ScaleModule (implies Identity)."""

from torch import Tensor
from torch.nn import Identity

from backpack_local.core.derivatives.basederivatives import BaseDerivatives
from backpack_local.custom_module.scale_module import ScaleModule


class ScaleModuleDerivatives(BaseDerivatives):
    """Derivatives of ScaleModule (implies Identity)."""

    def _jac_t_mat_prod(
        self,
        module: ScaleModule | Identity,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        mat: Tensor,
        subsampling: list[int] = None,
    ) -> Tensor:
        if isinstance(module, Identity):
            return mat
        else:
            return mat * module.weight
