"""
models.py — Definisi Arsitektur Deep Learning
==============================================
Model untuk klasifikasi sentimen e-commerce (3 kelas).
Arsitektur: BiLSTMClassifier (BiLSTM dua lapis dengan concat last hidden)
"""

import torch
import torch.nn as nn

from config import (
    VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_CLASSES
)

# ──────────────────────────────────────────────
# MODEL: BiLSTM
# ──────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    BiLSTM dua lapis untuk klasifikasi teks sentimen.

    Arsitektur:
        Input (token indices)
          → Embedding layer
          → Dropout
          → BiLSTM (2 layers)
          → Ambil hidden state terakhir (concat forward + backward)
          → Dropout
          → Fully Connected (hidden_dim*2 → num_classes)
    """

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim: int    = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        num_layers: int   = NUM_LAYERS,
        num_classes: int  = NUM_CLASSES,
        dropout: float    = DROPOUT,
        pad_idx: int      = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (batch, seq_len) — token indices
            lengths: (batch,) — panjang asli tiap urutan (sebelum padding)

        Returns:
            logits: (batch, num_classes)
        """
        embedded = self.dropout(self.embedding(x))   # (B, L, E)

        # Pack padded sequence agar LSTM tidak memproses padding
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: (num_layers*2, B, H)

        # Ambil hidden state lapisan terakhir (forward + backward)
        last_hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=1
        )  # (B, H*2)

        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


# ──────────────────────────────────────────────
# HELPER: Hitung jumlah parameter
# ──────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Mengembalikan jumlah parameter yang dapat dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)