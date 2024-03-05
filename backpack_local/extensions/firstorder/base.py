"""Base class for first order extensions."""

from torch.nn import Module

from backpack_local.extensions.backprop_extension import FAIL_WARN, BackpropExtension
from backpack_local.extensions.module_extension import ModuleExtension


class FirstOrderModuleExtension(ModuleExtension):
    """Base class for first order module extensions."""


class FirstOrderBackpropExtension(BackpropExtension):
    """Base backpropagation extension for first order."""

    def __init__(
        self,
        savefield: str,
        module_exts: dict[type[Module], ModuleExtension],
        fail_mode: str = FAIL_WARN,
        subsampling: list[int] = None,
    ):
        super().__init__(
            savefield, module_exts, fail_mode=fail_mode, subsampling=subsampling
        )

    def expects_backpropagation_quantities(self) -> bool:  # noqa: D102
        return False
