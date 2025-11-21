"""
Configuration settings for the PCA Face Recognition Demo.

This module loads configuration from environment variables defined in .env file.
All values must be defined in the .env file - no defaults are provided.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Any

# Load environment variables from .env file
load_dotenv()

# Helper function to get required environment variable
def get_env(key: str, cast_type: type = str) -> Any:
    """Get required environment variable with type casting."""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' is not set in .env file")

    if cast_type == int:
        return int(value)
    elif cast_type == float:
        return float(value)
    elif cast_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    else:
        return value

# Dataset configuration
DATASET_CONFIG = {
    "dataset_path": get_env("DATASET_PATH"),
    "expected_subjects": get_env("EXPECTED_SUBJECTS", int),
    "expected_images_per_subject": get_env("EXPECTED_IMAGES_PER_SUBJECT", int),
    "image_format": get_env("IMAGE_FORMAT"),
    "image_size": (
        get_env("IMAGE_WIDTH", int),
        get_env("IMAGE_HEIGHT", int)
    ),
}

# Default PCA parameters
PCA_CONFIG = {
    "default_n_components": get_env("DEFAULT_N_COMPONENTS", int),
    "min_n_components": get_env("MIN_N_COMPONENTS", int),
    "max_n_components": get_env("MAX_N_COMPONENTS", int),
    "default_distance_metric": get_env("DEFAULT_DISTANCE_METRIC"),
    "distance_metrics": ["Euclidean", "Cosine"],
}

# Training configuration
TRAINING_CONFIG = {
    "default_training_images_per_subject": get_env("DEFAULT_TRAINING_IMAGES_PER_SUBJECT", int),
    "min_training_images": get_env("MIN_TRAINING_IMAGES", int),
    "max_training_images": get_env("MAX_TRAINING_IMAGES", int),
}

# App configuration
APP_CONFIG = {
    "app_name": get_env("APP_NAME"),
    "page_icon": get_env("PAGE_ICON"),
    "layout": get_env("LAYOUT"),
    "default_port": get_env("DEFAULT_PORT", int),
}

# Get absolute paths
BASE_DIR = Path(__file__).parent
DATASET_CONFIG["dataset_path_abs"] = BASE_DIR / DATASET_CONFIG["dataset_path"]