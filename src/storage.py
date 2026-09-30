import os
import logging
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Set up professional logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

Base = declarative_base()


class ReviewRecord(Base):
    """
    SQLAlchemy ORM Model representing the database schema.
    This mirrors our Pydantic validation model.
    """

    __tablename__ = "validated_reviews"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Core Features
    clothing_id = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    title = Column(String, nullable=True)
    review_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)
    recommended_ind = Column(Integer, nullable=False)
    positive_feedback_count = Column(Integer, nullable=False)

    # Categorical Features
    division_name = Column(String, nullable=True)
    department_name = Column(String, nullable=True)
    class_name = Column(String, nullable=True)


def store_data(
    df: pd.DataFrame, db_dir: str = "data", db_name: str = "ecommerce.db"
) -> str:
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, db_name)

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    logging.info("Initializing database schema...")
    Base.metadata.create_all(engine)

    column_mapping = {
        "Clothing ID": "clothing_id",
        "Age": "age",
        "Title": "title",
        "Review Text": "review_text",
        "Rating": "rating",
        "Recommended IND": "recommended_ind",
        "Positive Feedback Count": "positive_feedback_count",
        "Division Name": "division_name",
        "Department Name": "department_name",
        "Class Name": "class_name",
    }

    if "Review Text" in df.columns:
        df = df.rename(columns=column_mapping)

    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].fillna("").astype(str)

    # Load the DataFrame into the SQLite database
    logging.info(
        f"Writing {len(df)} validated records to SQLite database at {db_path}..."
    )

    try:
        with sqlite3.connect(db_path) as conn:
            df.to_sql("validated_reviews", con=conn, if_exists="replace", index=False)
        logging.info("Data successfully stored in the database.")
    except Exception as e:
        logging.error(f"Failed to store data in the database: {e}")
        raise RuntimeError("Pipeline halted: Database storage failure.") from e

    return db_path


if __name__ == "__main__":
    # Test the storage logic
    logging.info("Testing storage module...")

    dummy_data = pd.DataFrame(
        [
            {
                "clothing_id": 767,
                "age": 33,
                "title": "Great fit!",
                "review_text": "I absolutely love this dress. It fits perfectly and the material is soft.",
                "rating": 5,
                "recommended_ind": 1,
                "positive_feedback_count": 0,
                "division_name": "Initmates",
                "department_name": "Intimate",
                "class_name": "Intimates",
            }
        ]
    )

    saved_db_path = store_data(dummy_data)
    print(f"Test complete. Database created at: {saved_db_path}")
