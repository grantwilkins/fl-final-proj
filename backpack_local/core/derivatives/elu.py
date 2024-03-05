"""Partial derivatives for the ELU activation function."""

from torch import Tensor, exp, le, ones_like, zeros_like
from torch.nn import ELU

from backpack_local.core.derivatives.elementwise import ElementwiseDerivatives
from backpack_local.utils.subsampling import subsample


class ELUDerivatives(ElementwiseDerivatives):
    """Implement first- and second-order partial derivatives of ELU."""

    def hessian_is_zero(self, module: ELU) -> bool:
        """`ELU''(x) ≠ 0`."""
        return False

    def df(
        self,
        module: ELU,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        subsampling: list[int] = None,
    ):
        """First ELU derivative: `ELU'(x) = alpha * e^x if x <= 0 else 1`."""
        input0 = subsample(module.input0, subsampling=subsampling)
        non_pos = le(input0, 0)

        result = ones_like(input0)
        result[non_pos] = module.alpha * exp(input0[non_pos])

        return result

    def d2f(self, module, g_inp, g_out):
        """Second ELU derivative: `ELU''(x) = alpha * e^x if x <= 0 else 0`."""
        non_pos = le(module.input0, 0)

        result = zeros_like(module.input0)
        result[non_pos] = module.alpha * exp(module.input0[non_pos])

        return result
