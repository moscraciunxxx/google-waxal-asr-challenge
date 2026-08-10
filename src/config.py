"""Shared configuration for the WAXAL ASR solution."""

from __future__ import annotations

from pathlib import Path

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Challenge languages (ISO 639-2 / HF config suffixes)
LANGUAGES = ("lin", "sna", "lug")
LANGUAGE_NAMES = {
    "lin": "Lingala",
    "sna": "Shona",
    "lug": "Luganda",
}
HF_DATASET = "google/WaxalNLP"
HF_CONFIGS = {lang: f"{lang}_asr" for lang in LANGUAGES}

# Open pretrained backbone (openly available, no closed APIs)
# Whisper is open (MIT). Fine-tuned jointly for Phase-1 and audio-only Phase-2.
DEFAULT_MODEL_ID = "openai/whisper-small"
# Competitive upgrade path
COMPETITIVE_MODEL_ID = "openai/whisper-medium"

# MMS supports lin/sna/lug natively (CC-BY-NC weights, openly downloadable)
MMS_MODEL_ID = "facebook/mms-1b-all"
MMS_LANG_MAP = {"lin": "lin", "sna": "sna", "lug": "lug"}

# Audio
TARGET_SR = 16_000
MAX_AUDIO_SECONDS = 30.0  # Whisper window; longer clips are chunked

# Training defaults
SEED = 42
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRAD_ACCUM = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
MAX_LABEL_LENGTH = 448
FP16 = False  # MPS prefers float32 / bf16 handling in train loop
BF16_MPS = False

# Metric weights (official Zindi multi-metric)
WER_WEIGHT = 0.5
CER_WEIGHT = 0.5

# CSV schema (Zindi-compatible)
ID_COL = "ID"
TARGET_COL = "Target"
LANG_COL = "language"
SPLIT_COL = "split"

# Explicit rule: never use HF/Zindi Phase-1 test gold for training/tuning/pseudo-labeling
FORBIDDEN_TRAIN_SPLITS = frozenset({"test"})

# Paths for reconstructed Zindi tables
TRAIN_CSV = DATA_DIR / "Train.csv"
TEST_CSV = DATA_DIR / "Test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "SampleSubmission.csv"
INDEX_CSV = DATA_DIR / "dataset_index.csv"
METADATA_CACHE = DATA_DIR / "hf_metadata"

# The Phase-2 portal set was expanded after the original 1,500-row release.
# Keep the historical template untouched for reproducibility, but expose the
# current workspace paths explicitly so generic Phase-1 validation cannot be
# accidentally used for a live Phase-2 upload.
PHASE2_DIR = DATA_DIR / "phase2"
PHASE2_SAMPLE_SUBMISSION_CSV = PHASE2_DIR / "SampleSubmission.csv"
PHASE2_TEST_CSV = PHASE2_DIR / "Test.csv"
PHASE2_NEW_AUDIO_DIR = PROJECT_ROOT / "newaudios"
