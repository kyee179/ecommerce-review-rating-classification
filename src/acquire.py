import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def acquire_data(
    dataset_identifier: str = "nicapotato/womens-ecommerce-clothing-reviews",
    dest_folder: str = "data",
) -> str:
    os.makedirs(dest_folder, exist_ok=True)
    expected_csv_path = os.path.join(
        dest_folder, "Womens Clothing E-Commerce Reviews.csv"
    )
    if os.path.exists(expected_csv_path):
        logging.info(
            f"Dataset already exists at {expected_csv_path}. Skipping API download."
        )
        return expected_csv_path

    logging.info(f"Attempting to download {dataset_identifier} via Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_identifier, path=dest_folder, unzip=True)
        logging.info("Dataset successfully downloaded and unzipped.")
        return expected_csv_path
    except Exception as e:
        logging.error(f"Kaggle API acquisition failed: {e}")
        raise RuntimeError(
            "Pipeline halted: Cannot proceed without training data."
        ) from e


if __name__ == "__main__":
    downloaded_file = acquire_data()
    print(f"Data ready for validation at: {downloaded_file}")
