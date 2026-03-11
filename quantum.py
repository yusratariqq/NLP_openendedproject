"""
Quantum-inspired fidelity scoring
"""
import numpy as np

class QuantumFidelityScorer:
    """Compute quantum-inspired semantic fidelity"""

    def __init__(self):
        pass

    def compute_fidelity(self, claim_embedding: np.ndarray,
                        evidence_embedding: np.ndarray) -> float:
        """
        Compute F = cos²(θ) between claim and evidence

        Args:
            claim_embedding: Claim embedding vector
            evidence_embedding: Evidence embedding vector

        Returns:
            Fidelity score (0 to 1)
        """
        # Normalize vectors
        u = claim_embedding / (np.linalg.norm(claim_embedding) + 1e-10)
        v = evidence_embedding / (np.linalg.norm(evidence_embedding) + 1e-10)

        # Cosine similarity
        cos_theta = np.dot(u, v)

        # Clip to valid range [-1, 1]
        cos_theta = np.clip(cos_theta, -1, 1)

        # Fidelity (squared cosine)
        fidelity = (cos_theta + 1) / 2

        return float(fidelity)

    def compute_fidelity_batch(self, claim_embedding: np.ndarray,
                               evidence_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute fidelity for multiple evidence pieces

        Args:
            claim_embedding: Single claim embedding
            evidence_embeddings: Array of evidence embeddings (n_evidence, dim)

        Returns:
            Array of fidelity scores
        """
        fidelities = []
        for evidence_emb in evidence_embeddings:
            f = self.compute_fidelity(claim_embedding, evidence_emb)
            fidelities.append(f)
        return np.array(fidelities)

    def average_fidelity(self, claim_embedding: np.ndarray,
                        evidence_embeddings: np.ndarray) -> float:
        """
        Compute average fidelity across all evidence

        Args:
            claim_embedding: Claim embedding
            evidence_embeddings: Evidence embeddings

        Returns:
            Average fidelity score
        """
        fidelities = self.compute_fidelity_batch(claim_embedding, evidence_embeddings)
        return float(np.mean(fidelities)) if len(fidelities) > 0 else 0.0
