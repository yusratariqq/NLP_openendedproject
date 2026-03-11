"""
Configuration for multilingual fact-verification system
"""
import os

class Config:

    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')
    INDEX_DIR = os.path.join(DATA_DIR, 'indices')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    SUPPORTED_LANGUAGES = ['ar', 'ta']
    LANGUAGE_NAMES = {
        'ar': 'Arabic',
        'ta': 'Tamil',
    }

    # Dataset settings
    FINEWIKI_SUBSET_SIZE = 50000
    FINEWEB2_SUBSET_SIZE = 80000
    XFACT_SIZE = None
    ENGLISH_SUBSET_SIZE  = 5000

    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/LaBSE"
    EMBEDDING_DIM = 768
    NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    # Stage 1: Embedding fine-tuning
    STAGE1_EPOCHS = 1
    STAGE1_BATCH_SIZE = 16
    STAGE1_LEARNING_RATE = 2e-5
    STAGE1_WARMUP_STEPS = 500
    STAGE1_NUM_PAIRS = 100000
    STAGE1_POSITIVE_RATIO = 0.5
    STAGE1_EVAL_STEPS = 1000
    STAGE1_SAVE_BEST = True
    STAGE1_VALIDATION_SPLIT = 0.1
    STAGE1_LOSS_TYPE = 'contrastive'

    # Stage 2: Threshold optimization ✅ renamed to match EvidenceGate
    STAGE2_NLI_RANGE        = [0.45, 0.50, 0.55, 0.60, 0.65]
    STAGE2_SIMILARITY_RANGE = [0.70, 0.75, 0.80, 0.85]
    STAGE2_SUPPORT_RANGE    = [0.30, 0.40, 0.50, 0.60]

    # Default thresholds ✅ renamed and recalibrated
    NLI_MIN        = 0.55
    SIMILARITY_MIN = 0.80
    SUPPORT_MIN    = 0.40

    # Retrieval settings
    TOP_K_RESULTS = 10
    FAISS_INDEX_TYPE = 'IndexFlatIP'

    # File paths
    EVIDENCE_INDEX_PATH       = os.path.join(INDEX_DIR, 'evidence_index.faiss')
    DOCUMENTS_PATH            = os.path.join(INDEX_DIR, 'documents.pkl')
    FINETUNED_MODEL_PATH      = os.path.join(MODEL_DIR, 'embeddings_finetuned')
    OPTIMIZED_THRESHOLDS_PATH = os.path.join(MODEL_DIR, 'optimized_thresholds.json')

    LOG_LEVEL = 'INFO'

    @classmethod
    def display(cls):
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Languages       : {', '.join(cls.SUPPORTED_LANGUAGES)}")
        print(f"FineWiki subset : {cls.FINEWIKI_SUBSET_SIZE} articles/lang")
        print(f"FineWeb2 subset : {cls.FINEWEB2_SUBSET_SIZE} docs/lang")
        print(f"Training pairs  : {cls.STAGE1_NUM_PAIRS} (pos/neg: {cls.STAGE1_POSITIVE_RATIO:.0%}/{1-cls.STAGE1_POSITIVE_RATIO:.0%})")
        print(f"Embedding model : {cls.EMBEDDING_MODEL}")
        print(f"NLI model       : {cls.NLI_MODEL}")
        print(f"Epochs          : {cls.STAGE1_EPOCHS}")
        print(f"Batch size      : {cls.STAGE1_BATCH_SIZE}")
        print(f"Thresholds      : nli={cls.NLI_MIN}, similarity={cls.SIMILARITY_MIN}, support={cls.SUPPORT_MIN}")
        print("=" * 60)
