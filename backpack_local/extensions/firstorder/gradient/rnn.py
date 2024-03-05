"""Contains GradRNN."""

from backpack_local.core.derivatives.lstm import LSTMDerivatives
from backpack_local.core.derivatives.rnn import RNNDerivatives
from backpack_local.extensions.firstorder.gradient.base import GradBaseModule


class GradRNN(GradBaseModule):
    """Extension for RNN, calculating gradient."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )


class GradLSTM(GradBaseModule):
    """Extension for LSTM, calculating gradient."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=LSTMDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )
