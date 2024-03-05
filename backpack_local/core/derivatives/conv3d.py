from backpack_local.core.derivatives.convnd import ConvNDDerivatives


class Conv3DDerivatives(ConvNDDerivatives):
    def __init__(self):
        super().__init__(N=3)
