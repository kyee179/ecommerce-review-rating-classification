import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import logging

# Set up professional logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BertRatingClassifier(nn.Module):
    """
    A PyTorch module wrapping a Hugging Face Transformer for rating prediction.
    We default to DistilBERT to handle CPU constraints effectively.
    """

    def __init__(
        self, model_name: str = "distilbert-base-uncased", num_classes: int = 5
    ):
        super(BertRatingClassifier, self).__init__()

        logging.info(f"Initializing {model_name} with {num_classes} output classes...")

        # Load the pre-trained Hugging Face model with a fresh classification head
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the BERT classifier.
        Args:
            input_ids: Tensor of token ids. Shape: (batch_size, max_seq_length)
            attention_mask: Tensor of mask values (0 or 1). Shape: (batch_size, max_seq_length)
        Returns:
            logits: Unnormalized prediction scores. Shape: (batch_size, num_classes)
        """
        # Pass the inputs through the transformer model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # We only need the raw logits for the PyTorch CrossEntropyLoss function
        return outputs.logits


if __name__ == "__main__":
    # Test the model initialization
    try:
        dummy_model = BertRatingClassifier()
        print("Model architecture successfully loaded:\n")
        print(dummy_model)
    except Exception as e:
        print(f"Failed to load model: {e}")
