"""Contains SGSBase, the base module for sum_grad_squared extension."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

from torch import Tensor
from torch.nn import Module

from backpack_local.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack_local.extensions.firstorder.base import FirstOrderModuleExtension

if TYPE_CHECKING:
    from backpack_local.extensions import SumGradSquared


class SGSBase(FirstOrderModuleExtension):
    """Base class for extensions calculating sum_grad_squared."""

    def __init__(self, derivatives: BaseParameterDerivatives, params: list[str] = None):
        """Initialization.

        For each parameter a function is initialized that is named like the parameter

        Args:
            derivatives: calculates the derivatives wrt parameters
            params: list of parameter names
        """
        self.derivatives: BaseParameterDerivatives = derivatives
        self.N_axis: int = 0
        for param_str in params:
            if not hasattr(self, param_str):
                setattr(self, param_str, self._make_param_function(param_str))
        super().__init__(params=params)

    def _make_param_function(
        self, param_str: str
    ) -> Callable[[SumGradSquared, Module, tuple[Tensor], tuple[Tensor], None], Tensor]:
        """Creates a function that calculates sum_grad_squared.

        Args:
            param_str: name of parameter

        Returns
        -------
            function that calculates sum_grad_squared
        """

        def param_function(
            ext: SumGradSquared,
            module: Module,
            g_inp: tuple[Tensor],
            g_out: tuple[Tensor],
            bpQuantities: None,
        ) -> Tensor:
            """Calculates sum_grad_squared with the help of derivatives object.

            Args:
                ext: extension that is used
                module: module that performed forward pass
                g_inp: input gradient tensors
                g_out: output gradient tensors
                bpQuantities: additional quantities for second order

            Returns
            -------
                sum_grad_squared
            """
            return (
                self.derivatives.param_mjp(
                    param_str, module, g_inp, g_out, g_out[0], sum_batch=False
                )
                ** 2
            ).sum(self.N_axis)

        return param_function
