import os
import torch
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    confusion_matrix,
)

matplotlib.use("Agg")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    logging.info(f"--- Calculating Metrics for {model_name} ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    mse = mean_squared_error(y_true, y_pred)

    logging.info(
        f"Accuracy:  {accuracy:.4f} | Macro F1:  {f1:.4f} | Precision: {precision:.4f} | Recall:    {recall:.4f}"
    )
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mse": mse,
    }


def save_confusion_matrix(y_true, y_pred, model_name="Model", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    y_true_shifted = [y + 1 for y in y_true]
    y_pred_shifted = [y + 1 for y in y_pred]

    cm = confusion_matrix(y_true_shifted, y_pred_shifted, labels=[1, 2, 3, 4, 5])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5],
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual Rating")
    plt.xlabel("Predicted Rating")

    file_path = os.path.join(
        output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def run_evaluation(model, test_loader, device, model_name="BERT_Classifier"):
    logging.info(f"Evaluating {model_name} on unseen test data...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            b_labels = batch["labels"].to(device)
            if "input_ids" in batch:
                logits = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
            else:
                logits = model(
                    batch["sequences"].to(device), batch["lengths"].to(device)
                )

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

    metrics = evaluate_predictions(all_labels, all_preds, model_name)
    save_confusion_matrix(all_labels, all_preds, model_name)
    return metrics
