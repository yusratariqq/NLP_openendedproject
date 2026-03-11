"""
Stage 1: Fine-tune embeddings using FineWeb2 data
"""
import sys
import os
try:
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_PATH = os.getcwd()

sys.path.append(BASE_PATH)

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

def create_training_pairs_balanced(documents, num_pairs=100000, positive_ratio=0.5):
    """
    Create balanced positive and negative training pairs

    Args:
        documents: List of document dictionaries
        num_pairs: Total number of pairs to create
        positive_ratio: Ratio of positive pairs (0.5 = 50/50 split)

    Returns:
        List of InputExample objects
    """
    import random

    num_positive = int(num_pairs * positive_ratio)
    num_negative = num_pairs - num_positive

    print(f"\nCreating {num_pairs} training pairs:")
    print(f"  Positive pairs: {num_positive}")
    print(f"  Negative pairs: {num_negative}")

    examples = []
    random.shuffle(documents)

    # ============================================
    # CREATE POSITIVE PAIRS
    # ============================================
    print("\n1. Creating positive pairs (similar texts)...")
    positive_count = 0

    for doc in tqdm(documents, desc="Positive pairs"):
        if positive_count >= num_positive:
            break

        text = doc.get('text', '')
        if len(text) < 200:
            continue

        # Split into chunks
        chunks = split_into_chunks(text, chunk_size=200, overlap=50)

        # Create pairs from same document (semantically similar)
        for i in range(len(chunks) - 1):
            if positive_count >= num_positive:
                break

            examples.append(InputExample(
                texts=[chunks[i], chunks[i + 1]],
                label=1.0  # Similar
            ))
            positive_count += 1

    print(f"✅ Created {positive_count} positive pairs")

    # ============================================
    # CREATE NEGATIVE PAIRS
    # ============================================
    print("\n2. Creating negative pairs (dissimilar texts)...")
    negative_count = 0

    # Filter documents by length for efficiency
    valid_docs = [d for d in documents if len(d.get('text', '')) >= 100]

    pbar = tqdm(total=num_negative, desc="Negative pairs")

    while negative_count < num_negative:
        # Pick two random documents
        doc1 = random.choice(valid_docs)
        doc2 = random.choice(valid_docs)

        # Ensure different documents
        text1 = doc1.get('text', '')
        text2 = doc2.get('text', '')

        if text1 == text2:
            continue

        # Take first 200 chars from each
        text1 = text1[:200].strip()
        text2 = text2[:200].strip()

        # Filter very short texts
        if len(text1) < 100 or len(text2) < 100:
            continue

        examples.append(InputExample(
            texts=[text1, text2],
            label=0.0  # Dissimilar
        ))
        negative_count += 1
        pbar.update(1)

    pbar.close()
    print(f"✅ Created {negative_count} negative pairs")

    # ============================================
    # SHUFFLE AND RETURN
    # ============================================
    random.shuffle(examples)

    # Verify counts
    pos_count = sum(1 for e in examples if e.label > 0.5)
    neg_count = sum(1 for e in examples if e.label < 0.5)

    print(f"\n✅ Total pairs created: {len(examples)}")
    print(f"   Positive: {pos_count} ({pos_count/len(examples)*100:.1f}%)")
    print(f"   Negative: {neg_count} ({neg_count/len(examples)*100:.1f}%)")

    return examples

def finetune_embeddings():
    """Fine-tune embedding model on FineWeb2 data with validation"""

    print("=" * 60)
    print("STAGE 1: FINE-TUNING EMBEDDINGS")
    print("=" * 60)

    Config.display()

    # ============================================
    # LOAD DATA
    # ============================================
    print("\nLoading FineWeb2 data...")
    fineweb2_data = load_data('fineweb2_data.pkl')

    # Combine all documents
    all_documents = []
    for lang in Config.SUPPORTED_LANGUAGES:
        docs = fineweb2_data.get(lang, [])
        all_documents.extend(docs)
        print(f"  {lang}: {len(docs)} documents")

    print(f"Total documents: {len(all_documents)}")

    # ============================================
    # LOAD BASE MODEL
    # ============================================
    print(f"\nLoading base model: {Config.EMBEDDING_MODEL}")
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    print(f"✅ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ============================================
    # CREATE TRAINING PAIRS
    # ============================================
    training_pairs = create_training_pairs_balanced(
        all_documents,
        num_pairs=Config.STAGE1_NUM_PAIRS,
        positive_ratio=Config.STAGE1_POSITIVE_RATIO
    )

    # ============================================
    # SPLIT TRAIN/VALIDATION
    # ============================================
    print(f"\nSplitting data (train/val: {1-Config.STAGE1_VALIDATION_SPLIT:.0%}/{Config.STAGE1_VALIDATION_SPLIT:.0%})...")

    split_idx = int(len(training_pairs) * (1 - Config.STAGE1_VALIDATION_SPLIT))
    train_pairs = training_pairs[:split_idx]
    val_pairs = training_pairs[split_idx:]

    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Validation pairs: {len(val_pairs)}")

    # ============================================
    # PREPARE DATALOADERS
    # ============================================
    print("\nPreparing dataloaders...")
    train_dataloader = DataLoader(
        train_pairs,
        shuffle=True,
        batch_size=Config.STAGE1_BATCH_SIZE
    )

    # ============================================
    # DEFINE LOSS FUNCTION
    # ============================================
    print(f"\nSetting up loss function: {Config.STAGE1_LOSS_TYPE}")

    if Config.STAGE1_LOSS_TYPE == 'contrastive':
        train_loss = losses.ContrastiveLoss(model)
        print("✅ Using ContrastiveLoss (better for positive/negative pairs)")
    else:
        train_loss = losses.CosineSimilarityLoss(model)
        print("✅ Using CosineSimilarityLoss")

    # ============================================
    # SETUP EVALUATOR
    # ============================================
    print("\nSetting up validation evaluator...")
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

    # Create evaluation data (use subset for speed)
    eval_subset_size = min(500, len(val_pairs))
    sentences1 = [pair.texts[0] for pair in val_pairs[:eval_subset_size]]
    sentences2 = [pair.texts[1] for pair in val_pairs[:eval_subset_size]]
    scores = [pair.label for pair in val_pairs[:eval_subset_size]]

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1,
        sentences2,
        scores,
        name='validation',
        show_progress_bar=True
    )

    print(f"✅ Evaluator ready with {eval_subset_size} validation pairs")

    # ============================================
    # TRAINING
    # ============================================
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {Config.STAGE1_EPOCHS}")
    print(f"Batch size: {Config.STAGE1_BATCH_SIZE}")
    print(f"Learning rate: {Config.STAGE1_LEARNING_RATE}")
    print(f"Warmup steps: {Config.STAGE1_WARMUP_STEPS}")
    print(f"Evaluation every: {Config.STAGE1_EVAL_STEPS} steps")
    print(f"Total training steps: {len(train_dataloader) * Config.STAGE1_EPOCHS}")
    print("=" * 60)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=Config.STAGE1_EPOCHS,
        warmup_steps=Config.STAGE1_WARMUP_STEPS,
        evaluator=evaluator,
        evaluation_steps=Config.STAGE1_EVAL_STEPS,
        output_path=Config.FINETUNED_MODEL_PATH,
        save_best_model=Config.STAGE1_SAVE_BEST,
        checkpoint_path='checkpoints/stage1',  # ✅ saves every epoch
    checkpoint_save_steps=5625,
        show_progress_bar=True,
        optimizer_params={'lr': Config.STAGE1_LEARNING_RATE}
    )

    # ============================================
    # POST-TRAINING VALIDATION
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - FINAL VALIDATION")
    print("=" * 60)

    # Load best model
    if Config.STAGE1_SAVE_BEST and os.path.exists(Config.FINETUNED_MODEL_PATH):
        print(f"Loading best model from: {Config.FINETUNED_MODEL_PATH}")
        model = SentenceTransformer(Config.FINETUNED_MODEL_PATH)

    # Final evaluation
    print("\nRunning final validation...")
    final_score = evaluator(model)
    print(f"Final validation score: {final_score}")

    # Test cross-lingual similarity
    print("\n" + "=" * 60)
    print("TESTING CROSS-LINGUAL EMBEDDINGS")
    print("=" * 60)

    # Test with sample texts
    test_samples = {
        'ar': "هذا كتاب جيد",  # "This is a good book" in Arabic
        'ta': "இது ஒரு நல்ல புத்தகம்",  # "This is a good book" in Tamil
        'en': "This is a good book"
    }

    embeddings = {}
    for lang, text in test_samples.items():
        embeddings[lang] = model.encode(text)
        print(f"{lang}: {text[:50]}...")

    # Compute similarities
    import numpy as np
    print("\nCross-lingual similarities:")
    print(f"  Arabic-Tamil: {np.dot(embeddings['ar'], embeddings['ta']):.3f}")
    print(f"  Arabic-English: {np.dot(embeddings['ar'], embeddings['en']):.3f}")
    print(f"  Tamil-English: {np.dot(embeddings['ta'], embeddings['en']):.3f}")
    print("\n(Should be > 0.6 for good cross-lingual alignment)")

    # ============================================
    # SAVE FINAL MODEL
    # ============================================
    print("\n" + "=" * 60)
    print("✅ STAGE 1 COMPLETE")
    print("=" * 60)
    print(f"Fine-tuned model saved to: {Config.FINETUNED_MODEL_PATH}")
    print(f"Model is ready for Stage 2 (threshold optimization)")
    print("=" * 60)

if __name__ == "__main__":
    finetune_embeddings()


