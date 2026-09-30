import os
import pandas as pd
import logging
import sqlite3
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_split_data(db_path: str = "data/ecommerce.db", target_col: str = "rating"):
    logging.info("Loading validated data from SQLite...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}.")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM validated_reviews", con=conn)

    # ==================================================
    # 1. Standard Text (For LSTM and Standard BERT)
    # ==================================================
    df["title"] = df["title"].fillna("No Title")
    df["standard_text"] = df["title"] + " - " + df["review_text"]

    # ==================================================
    # 2. Early Fusion Text (For Multimodal BERT)
    # ==================================================
    df["department_name"] = df["department_name"].fillna("Unknown Department")
    df["recommended_str"] = df["recommended_ind"].map({1: "Yes", 0: "No"})

    df["fused_text"] = (
        "Reviewer Age: "
        + df["age"].astype(str)
        + ". "
        + "Recommends product: "
        + df["recommended_str"]
        + ". "
        + "Department: "
        + df["department_name"]
        + ". "
        + "Title: "
        + df["title"]
        + ". "
        + "Review: "
        + df["review_text"]
    )

    df = df[["standard_text", "fused_text", target_col]]

    # --- PyTorch Label Shifting (1-5 -> 0-4) ---
    df[target_col] = df[target_col] - 1

    # --- Downsampling ---
    min_class_size = df[target_col].value_counts().min()
    df_balanced = df.groupby(target_col).sample(n=min_class_size, random_state=42)

    # --- Splitting ---
    train_df, temp_df = train_test_split(
        df_balanced, test_size=0.20, random_state=42, stratify=df_balanced[target_col]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df[target_col]
    )

    return train_df, val_df, test_df
