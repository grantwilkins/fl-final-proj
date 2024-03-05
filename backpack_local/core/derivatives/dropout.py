"""Partial derivatives for the dropout layer."""

from torch import Tensor, eq, ones_like
from torch.nn import Dropout

from backpack_local.core.derivatives.elementwise import ElementwiseDerivatives
from backpack_local.utils.subsampling import subsample


class DropoutDerivatives(ElementwiseDerivatives):
    """Derivatives for the Dropout module."""

    def hessian_is_zero(self, module: Dropout) -> bool:
        """``Dropout''(x) = 0``.

        Args:
            module: dropout module

        Returns
        -------
            whether hessian is zero
        """
        return True

    def df(
        self,
        module: Dropout,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        subsampling: list[int] = None,
    ) -> Tensor:
        output = subsample(module.output, subsampling=subsampling)
        if module.training:
            scaling = 1 / (1 - module.p)
            mask = 1 - eq(output, 0.0).to(output.dtype)
            return mask * scaling
        else:
            return ones_like(output)
