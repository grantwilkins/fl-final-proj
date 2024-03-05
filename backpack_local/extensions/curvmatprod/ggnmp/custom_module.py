"""Module extensions for custom properties of GGNMPBaseModule."""

from backpack_local.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack_local.core.derivatives.sum_module import SumModuleDerivatives
from backpack_local.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPScaleModule(GGNMPBase):
    """GGNMP extension for ScaleModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())


class GGNMPSumModule(GGNMPBase):
    """GGNMP extension for SumModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=SumModuleDerivatives())
