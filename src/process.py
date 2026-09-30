import logging
import torch
from collections import Counter
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def tokenize_for_bert(
    texts: list, model_name: str = "distilbert-base-uncased", max_length: int = 128
):
    logging.info(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_data = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded_data["input_ids"], encoded_data["attention_mask"]


def tokenize_for_lstm(
    train_texts: list,
    val_texts: list,
    test_texts: list,
    max_length: int = 128,
    max_vocab: int = 10000,
):
    """
    Builds a custom vocabulary from the training set and integer-encodes sequences for the LSTM.
    """
    logging.info("Building custom vocabulary for LSTM...")

    # 1. Simple whitespace tokenizer
    train_tokenized = [str(text).lower().split() for text in train_texts]
    val_tokenized = [str(text).lower().split() for text in val_texts]
    test_tokenized = [str(text).lower().split() for text in test_texts]

    # 2. Build Vocabulary
    word_counts = Counter(word for sentence in train_tokenized for word in sentence)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    def encode_and_pad(tokenized_texts):
        sequences = []
        lengths = []
        for tokens in tokenized_texts:
            seq = [vocab.get(word, vocab["<unk>"]) for word in tokens]
            length = max(1, min(len(seq), max_length))
            lengths.append(length)

            if len(seq) < max_length:
                seq = seq + [vocab["<pad>"]] * (max_length - len(seq))
            else:
                seq = seq[:max_length]
            sequences.append(seq)

        return torch.tensor(sequences, dtype=torch.long), torch.tensor(
            lengths, dtype=torch.long
        )

    logging.info("Encoding sequences for LSTM...")
    train_seqs, train_lens = encode_and_pad(train_tokenized)
    val_seqs, val_lens = encode_and_pad(val_tokenized)
    test_seqs, test_lens = encode_and_pad(test_tokenized)

    return (train_seqs, train_lens), (val_seqs, val_lens), (test_seqs, test_lens), vocab
