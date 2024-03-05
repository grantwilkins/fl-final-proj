"""BatchGrad extension for Embedding."""

from backpack_local.core.derivatives.embedding import EmbeddingDerivatives
from backpack_local.extensions.firstorder.batch_grad.batch_grad_base import (
    BatchGradBase,
)


class BatchGradEmbedding(BatchGradBase):
    """BatchGrad extension for Embedding."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=EmbeddingDerivatives(), params=["weight"])
