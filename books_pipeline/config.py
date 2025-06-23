"""
config.py - Configuration file for books processing pipeline
"""

# Input paths
INPUT_CSV_DIR = "./books_pipeline/input/csv"
INPUT_PDF_DIR = "./books_pipeline/input/pdfs"

# Intermediate paths
OCR_OUTPUT_DIR = "./books_pipeline/intermediate/ocr"
CLEANED_OUTPUT_DIR = "./books_pipeline/intermediate/cleaned"
EMBEDDINGS_DIR = "./books_pipeline/intermediate/embeddings"

# Final output paths
OUTPUT_DIR = "./books_pipeline/output"
EMBEDDINGS_OUTPUT_DIR = "./books_pipeline/output/embeddings"
IDS_OUTPUT_DIR = "./books_pipeline/output/ids"
INDEXES_OUTPUT_DIR = "./books_pipeline/output/indexes"

# Processing settings
CHUNK_MAX_LEN = 350
CHUNK_MIN_LEN = 175
CHUNK_OVERLAP = 50

# Multiprocessing
CPU_CORES = None  # None = auto-detect, or set specific number

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Database settings (for Django integration)
DATABASE_BATCH_SIZE = 100
