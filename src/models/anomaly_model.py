"""
LSTM Autoencoder for time-series anomaly detection.

The encoder compresses an input sequence to a fixed-size hidden state.
The decoder reconstructs the sequence from that hidden state. High
reconstruction error on unseen windows indicates anomalous behaviour.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.

    Architecture:
      Encoder  — LSTM consuming (batch, seq_len, input_dim), output: final hidden state
      Decoder  — repeats hidden state across seq_len, LSTM + Linear to reconstruct input
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            Reconstruction of the same shape as x.
        """
        seq_len = x.size(1)

        # Encode: run the full sequence through the encoder LSTM.
        # We only keep the final hidden state — it is the compressed representation.
        _, (hidden, cell) = self.encoder(x)

        # The decoder receives the encoder's final hidden state repeated across
        # every timestep, giving it the context to reconstruct each step.
        # hidden shape: (num_layers, batch, hidden_dim)
        # Take the last layer's hidden vector as the seed for the decoder input.
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        # decoder_input shape: (batch, seq_len, hidden_dim)

        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        # decoded shape: (batch, seq_len, hidden_dim)

        reconstruction = self.output_layer(decoded)
        # reconstruction shape: (batch, seq_len, input_dim)

        return reconstruction
