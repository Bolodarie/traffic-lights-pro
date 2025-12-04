import os
from pathlib import Path

# Caminhos Base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Hiperparâmetros Default
DEFAULT_IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
