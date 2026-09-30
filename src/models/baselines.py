import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==========================================
# 1. Simple Non-Neural Baseline
# ==========================================


def get_logistic_regression_baseline():
    """
    Creates a simple text classification pipeline using TF-IDF and Logistic Regression.
    This serves as the absolute foundational baseline to prove that deep learning
    (LSTM/BERT) is actually necessary for this specific NLP task.
    """
    pipeline = Pipeline(
        [
            # Convert raw text into a matrix of TF-IDF features
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            # Train a Logistic Regression classifier
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return pipeline


# ==========================================
# 2. Deep Learning Baseline (LSTM)
# ==========================================


class LSTMBaseline(nn.Module):
    """
    PyTorch implementation of an LSTM for sequence classification.
    Expects integer-encoded sequences (not raw text) as input.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super(LSTMBaseline, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, text_lengths):
        """
        Forward pass of the LSTM.
        Args:
            text: Tensor containing integer-encoded sequences. Shape: (batch_size, max_seq_length)
            text_lengths: Tensor containing the true, unpadded lengths of each sequence.
        """
        embedded = self.dropout(self.embedding(text))

        # Pack the padded sequence to ensure the LSTM ignores padding tokens
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        final_hidden_state = self.dropout(hidden[-1, :, :])

        logits = self.fc(final_hidden_state)

        return logits
