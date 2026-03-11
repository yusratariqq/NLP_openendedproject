"""
Hybrid multilingual evidence retrieval
"""
import faiss
import numpy as np
import pickle
from typing import List, Dict
import os

class EvidenceRetriever:
    """Retrieve relevant evidence using FAISS index"""

    def __init__(self, index_path: str, documents_path: str, embedding_generator):
        """
        Args:
            index_path: Path to FAISS index
            documents_path: Path to documents pickle file
            embedding_generator: EmbeddingGenerator instance
        """
        self.embedder = embedding_generator

        # Load index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"✅ Loaded FAISS index from {index_path}")
        else:
            raise FileNotFoundError(f"Index not found: {index_path}")

        # Load documents
        if os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"✅ Loaded {len(self.documents)} documents")
        else:
            raise FileNotFoundError(f"Documents not found: {documents_path}")

    def retrieve(self, claim: str, lang: str = None, top_k: int = 10) -> List[Dict]:
        """
        Retrieve relevant evidence documents

        Args:
            claim: Claim text to verify
            lang: Language of claim (optional)
            top_k: Number of results to retrieve

        Returns:
            List of evidence dictionaries with scores
        """
        # Encode claim
        query_embedding = self.embedder.encode(claim, normalize=True)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'title': doc.get('title', 'Untitled'),
                    'text': doc.get('text', ''),
                    'language': doc.get('language', 'unknown'),
                    'similarity': float(dist)
                })

        return results

    def retrieve_by_language(self, claim: str, target_lang: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve evidence prioritizing specific language

        Args:
            claim: Claim text
            target_lang: Preferred language
            top_k: Number of results

        Returns:
            List of evidence dictionaries
        """
        # Retrieve more than needed
        candidates = self.retrieve(claim, top_k=top_k * 3)

        # Filter by language
        same_lang = [doc for doc in candidates if doc['language'] == target_lang]
        other_lang = [doc for doc in candidates if doc['language'] != target_lang]

        # Combine: prioritize same language
        results = same_lang[:top_k]
        if len(results) < top_k:
            results.extend(other_lang[:top_k - len(results)])

        return results[:top_k]
