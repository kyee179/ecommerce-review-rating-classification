import os
import logging
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from src.acquire import acquire_data
from src.validate import clean_and_validate
from src.storage import store_data
from src.build_dataset import load_and_split_data

from src.process import tokenize_for_bert, tokenize_for_lstm
from src.models.baselines import LSTMBaseline
from src.models.bert_classifier import BertRatingClassifier
from src.train import set_seed, ReviewDataset, LSTMDataset, train_model
from src.evaluate import run_evaluation

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_variance_boxplot(results_df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="F1-Score", data=results_df, palette="Set2")
    sns.stripplot(
        x="Model", y="F1-Score", data=results_df, color=".25", size=8, jitter=True
    )
    plt.title("Model Performance Variance Across Multiple Random Seeds", fontsize=14)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    file_path = os.path.join(output_dir, "model_variance_boxplot.png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    logging.info(f"Variance box plot successfully saved to {file_path}")


def main():
    logging.info("Starting DLNLP Pipeline...")

    seeds = [42, 123, 999]
    num_epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_results = []

    try:
        # Phase 1: Data Pipeline
        raw_csv_path = acquire_data()
        clean_df = clean_and_validate(raw_csv_path)
        db_path = store_data(clean_df)
        train_df, val_df, test_df = load_and_split_data(db_path=db_path)
        batch_size = 16

        # Phase 2: Tokenization
        lstm_train, lstm_val, lstm_test, vocab = tokenize_for_lstm(
            train_df["standard_text"].tolist(),
            val_df["standard_text"].tolist(),
            test_df["standard_text"].tolist(),
        )
        std_train_seqs, std_train_masks = tokenize_for_bert(
            train_df["standard_text"].tolist()
        )
        std_val_seqs, std_val_masks = tokenize_for_bert(
            val_df["standard_text"].tolist()
        )
        std_test_seqs, std_test_masks = tokenize_for_bert(
            test_df["standard_text"].tolist()
        )

        fus_train_seqs, fus_train_masks = tokenize_for_bert(
            train_df["fused_text"].tolist()
        )
        fus_val_seqs, fus_val_masks = tokenize_for_bert(val_df["fused_text"].tolist())
        fus_test_seqs, fus_test_masks = tokenize_for_bert(
            test_df["fused_text"].tolist()
        )

        # Phase 3: The Experiment Loop
        for current_seed in seeds:
            logging.info(
                f"\n{'='*50}\nCOMMENCING EXPERIMENT WITH SEED: {current_seed}\n{'='*50}"
            )
            set_seed(current_seed)

            # --- MODEL 1: LSTM BASELINE ---
            logging.info("--- Training LSTM Baseline ---")
            lstm_train_loader = DataLoader(
                LSTMDataset(
                    lstm_train[0], lstm_train[1], train_df["rating"].tolist()
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            lstm_val_loader = DataLoader(
                LSTMDataset(lstm_val[0], lstm_val[1], val_df["rating"].tolist()),
                batch_size=batch_size,
                shuffle=False,
            )
            lstm_test_loader = DataLoader(
                LSTMDataset(lstm_test[0], lstm_test[1], test_df["rating"].tolist()),
                batch_size=batch_size,
                shuffle=False,
            )

            lstm_model = LSTMBaseline(
                vocab_size=len(vocab), embed_dim=128, hidden_dim=256, num_classes=5
            )
            trained_lstm = train_model(
                model=lstm_model,
                train_loader=lstm_train_loader,
                val_loader=lstm_val_loader,
                epochs=num_epochs,
                lr=1e-3,
            )
            lstm_metrics = run_evaluation(
                trained_lstm,
                lstm_test_loader,
                device,
                f"LSTM_Baseline_Seed_{current_seed}",
            )
            experiment_results.append(
                {
                    "Model": "LSTM Baseline",
                    "F1-Score": lstm_metrics["f1"],
                    "Seed": current_seed,
                }
            )

            # --- MODEL 2: STANDARD DISTILBERT ---
            logging.info("--- Training Standard DistilBERT ---")
            std_bert_train_loader = DataLoader(
                ReviewDataset(
                    std_train_seqs, std_train_masks, train_df["rating"].tolist()
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            std_bert_val_loader = DataLoader(
                ReviewDataset(
                    std_val_seqs, std_val_masks, val_df["rating"].tolist()
                ),
                batch_size=batch_size,
                shuffle=False,
            )
            std_bert_test_loader = DataLoader(
                ReviewDataset(
                    std_test_seqs, std_test_masks, test_df["rating"].tolist()
                ),
                batch_size=batch_size,
                shuffle=False,
            )

            std_bert_model = BertRatingClassifier(num_classes=5)
            trained_std_bert = train_model(
                model=std_bert_model,
                train_loader=std_bert_train_loader,
                val_loader=std_bert_val_loader,
                epochs=num_epochs,
                lr=2e-5,
            )
            std_bert_metrics = run_evaluation(
                trained_std_bert,
                std_bert_test_loader,
                device,
                f"Standard_DistilBERT_Seed_{current_seed}",
            )
            experiment_results.append(
                {
                    "Model": "Standard DistilBERT",
                    "F1-Score": std_bert_metrics["f1"],
                    "Seed": current_seed,
                }
            )

            # --- MODEL 3: EARLY FUSION DISTILBERT ---
            logging.info("--- Training Early Fusion DistilBERT ---")
            fus_bert_train_loader = DataLoader(
                ReviewDataset(
                    fus_train_seqs, fus_train_masks, train_df["rating"].tolist()
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            fus_bert_val_loader = DataLoader(
                ReviewDataset(fus_val_seqs, fus_val_masks, val_df["rating"].tolist()),
                batch_size=batch_size,
                shuffle=False,
            )
            fus_bert_test_loader = DataLoader(
                ReviewDataset(
                    fus_test_seqs, fus_test_masks, test_df["rating"].tolist()
                ),
                batch_size=batch_size,
                shuffle=False,
            )

            fus_bert_model = BertRatingClassifier(num_classes=5)

            # Pass dynamic num_epochs here
            trained_fus_bert = train_model(
                model=fus_bert_model,
                train_loader=fus_bert_train_loader,
                val_loader=fus_bert_val_loader,
                epochs=num_epochs,
                lr=2e-5,
            )

            fus_bert_metrics = run_evaluation(
                trained_fus_bert,
                fus_bert_test_loader,
                device,
                f"Early_Fusion_DistilBERT_Seed_{current_seed}",
            )
            experiment_results.append(
                {
                    "Model": "Early Fusion DistilBERT",
                    "F1-Score": fus_bert_metrics["f1"],
                    "Seed": current_seed,
                }
            )

        # Phase 4: Final Aggregation and Plotting
        logging.info("Runs complete. Processing results...")
        results_df = pd.DataFrame(experiment_results)
        print("\n--- Final Experimental Results ---")
        print(results_df.to_string(index=False))

        save_variance_boxplot(results_df, output_dir="plots")

        logging.info("Pipeline execution completed successfully!")

    except Exception as e:
        logging.critical(f"Pipeline failed critically: {e}")
        raise


if __name__ == "__main__":
    main()
