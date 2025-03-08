import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_NAME = "sohamvaidya1627/sign-varia"
DATASET_PATH = "data/"

def download_dataset():
    """Downloads the dataset using Kaggle API with environment variables."""
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        print("❌ Kaggle API credentials not found! Set KAGGLE_USERNAME and KAGGLE_KEY as environment variables.")
        return

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    print("📥 Downloading dataset...")
    api.dataset_download_files(DATASET_NAME, path=DATASET_PATH, unzip=True)
    print(f"✅ Dataset downloaded and extracted to `{DATASET_PATH}`.")

if __name__ == "__main__":
    download_dataset()
