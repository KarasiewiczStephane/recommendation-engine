"""MovieLens dataset download and extraction utilities.

Handles downloading the MovieLens 100K dataset from GroupLens,
extracting the archive, and verifying the data integrity.
"""

import urllib.request
import zipfile
from pathlib import Path

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_movielens(dest_path: str | None = None) -> Path:
    """Download and extract the MovieLens 100K dataset.

    Downloads the dataset zip archive if not already present, extracts it
    to the destination directory, and cleans up the archive file.

    Args:
        dest_path: Directory to store the extracted data. Defaults to the
            configured raw_path from config.yaml.

    Returns:
        Path to the extracted ml-100k directory.

    Raises:
        urllib.error.URLError: If the download fails.
        zipfile.BadZipFile: If the downloaded archive is corrupt.
    """
    if dest_path is None:
        dest_path = config["data"]["raw_path"]

    url = config["data"]["movielens_url"]
    dest = Path(dest_path)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ml-100k.zip"
    extracted_path = dest / "ml-100k"

    if not extracted_path.exists():
        logger.info("Downloading MovieLens 100K from %s", url)
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)

        zip_path.unlink()
        logger.info("Extraction complete at %s", extracted_path)
    else:
        logger.info("Dataset already exists at %s", extracted_path)

    return extracted_path
