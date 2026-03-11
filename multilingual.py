"""
Multilingual semantic embedding generation
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class EmbeddingGenerator:
    """Generate multilingual semantic embeddings"""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: Name/path of sentence transformer model
        """
        print(f"Loading embedding model: {model_name}")

        if os.path.exists(model_name):
            # Load from local path
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded fine-tuned model from {model_name}")
        else:
            # Load from HuggingFace
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded pre-trained model")

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for single text

        Args:
            text: Input text
            normalize: Whether to L2-normalize the embedding

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embedding

    def encode_batch(self, texts: list, batch_size: int = 32,
                     normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress
        )
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
