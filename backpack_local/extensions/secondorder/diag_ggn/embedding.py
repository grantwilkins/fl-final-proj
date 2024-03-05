"""DiagGGN extension for Embedding."""

from backpack_local.core.derivatives.embedding import EmbeddingDerivatives
from backpack_local.extensions.secondorder.diag_ggn.diag_ggn_base import (
    DiagGGNBaseModule,
)


class DiagGGNEmbedding(DiagGGNBaseModule):
    """DiagGGN extension of Embedding."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=EmbeddingDerivatives(), params=["weight"], sum_batch=True
        )


class BatchDiagGGNEmbedding(DiagGGNBaseModule):
    """DiagGGN extension of Embedding."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=EmbeddingDerivatives(), params=["weight"], sum_batch=False
        )
