import pandas as pd
import logging
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EcommerceReview(BaseModel):
    """
    Pydantic Object Model for a single E-Commerce Review.
    Enforces strict schema validation for our NLP pipeline.
    """

    clothing_id: int = Field(..., alias="Clothing ID")
    age: int = Field(..., ge=1, le=120, alias="Age")
    title: Optional[str] = Field(default=None, alias="Title")

    # NLP specific validation: text must exist and be somewhat meaningful
    review_text: str = Field(..., min_length=10, alias="Review Text")

    # Rating bounds: 1 to 5 stars
    rating: int = Field(..., ge=1, le=5, alias="Rating")

    recommended_ind: int = Field(..., ge=0, le=1, alias="Recommended IND")
    positive_feedback_count: int = Field(..., ge=0, alias="Positive Feedback Count")

    division_name: Optional[str] = Field(default=None, alias="Division Name")
    department_name: Optional[str] = Field(default=None, alias="Department Name")
    class_name: Optional[str] = Field(default=None, alias="Class Name")

    @field_validator("age", "clothing_id", "rating", mode="before")
    def check_no_decimals(cls, value):
        """Custom validator to ensure no decimal values are passed as integers."""
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Decimals are not allowed. Received: {value}")
        return int(value)


def clean_and_validate(raw_csv_path: str) -> pd.DataFrame:
    """
    Loads raw data, performs base cleaning (handling missing text),
    and applies strict Pydantic schema validation.
    """
    logging.info(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)

    # 1. Base Cleaning: Drop rows where the critical NLP feature (Review Text) is missing
    initial_count = len(df)
    df = df.dropna(subset=["Review Text"])
    logging.info(f"Dropped {initial_count - len(df)} rows missing 'Review Text'.")

    # 2. Pydantic Validation
    logging.info("Applying Pydantic schema validation...")
    valid_records = []
    error_count = 0

    records = df.to_dict(orient="records")

    for record in records:
        try:
            validated_record = EcommerceReview(**record)

            valid_records.append(validated_record.model_dump(by_alias=True))

        except ValidationError as e:
            error_count += 1

    logging.info(
        f"Validation complete. Kept {len(valid_records)} valid rows. Discarded {error_count} invalid rows."
    )

    clean_df = pd.DataFrame(valid_records)
    return clean_df


if __name__ == "__main__":
    test_file_path = "data/Womens Clothing E-Commerce Reviews.csv"
    try:
        clean_data = clean_and_validate(test_file_path)
        print("\nCleaned Data Preview:")
        print(clean_data[["Age", "Rating", "Review Text"]].head())
    except FileNotFoundError:
        print(f"Test file not found at {test_file_path}. Run acquire.py first!")
