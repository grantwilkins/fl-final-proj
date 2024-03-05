"""Contains Base class for batch_l2_grad."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

from torch import Tensor
from torch.nn import Module

from backpack_local.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack_local.extensions.firstorder.base import FirstOrderModuleExtension

if TYPE_CHECKING:
    from backpack_local.extensions import BatchL2Grad


class BatchL2Base(FirstOrderModuleExtension):
    """BaseExtension for batch_l2."""

    def __init__(self, params: list[str], derivatives: BaseParameterDerivatives = None):
        """Initialization.

        If derivatives object is provided initializes methods that compute batch_l2.
        If there is an existent method in a child class it is not overwritten.

        Args:
            params: parameter names
            derivatives: derivatives object. Defaults to None.
        """
        if derivatives is not None:
            self.derivatives: BaseParameterDerivatives = derivatives
            for param_str in params:
                if not hasattr(self, param_str):
                    setattr(self, param_str, self._make_param_function(param_str))
        super().__init__(params=params)

    def _make_param_function(
        self, param_str: str
    ) -> Callable[[BatchL2Grad, Module, tuple[Tensor], tuple[Tensor], None], Tensor]:
        """Creates a function that calculates batch_l2.

        Args:
            param_str: name of parameter

        Returns
        -------
            function that calculates batch_l2
        """

        def param_function(
            ext: BatchL2Grad,
            module: Module,
            g_inp: tuple[Tensor],
            g_out: tuple[Tensor],
            bpQuantities: None,
        ) -> Tensor:
            """Calculates batch_l2 with the help of derivatives object.

            Args:
                ext: extension that is used
                module: module that performed forward pass
                g_inp: input gradient tensors
                g_out: output gradient tensors
                bpQuantities: additional quantities for second order

            Returns
            -------
                batch_l2
            """
            param_dims: list[int] = list(range(1, 1 + getattr(module, param_str).dim()))
            return (
                self.derivatives.param_mjp(
                    param_str, module, g_inp, g_out, g_out[0], sum_batch=False
                )
                ** 2
            ).sum(param_dims)

        return param_function
