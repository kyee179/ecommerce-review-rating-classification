import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
import numpy as np
import random
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ReviewDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class LSTMDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sequences": self.sequences[idx],
            "lengths": self.lengths[idx],
            "labels": self.labels[idx],
        }


def train_model(model, train_loader, val_loader, epochs=5, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        logging.info(f"--- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            b_labels = batch["labels"].to(device)
            model.zero_grad()

            if "input_ids" in batch:
                logits = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
            else:
                logits = model(
                    batch["sequences"].to(device), batch["lengths"].to(device)
                )

            loss = criterion(logits, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        correct_predictions = 0
        with torch.no_grad():
            for batch in val_loader:
                b_labels = batch["labels"].to(device)
                if "input_ids" in batch:
                    logits = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                    )
                else:
                    logits = model(
                        batch["sequences"].to(device), batch["lengths"].to(device)
                    )

                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == b_labels).item()

        val_accuracy = correct_predictions / len(val_loader.dataset)
        logging.info(f"Validation Accuracy: {val_accuracy:.4f}\n")

    return model
