"""
NLI-based reliability scoring
Replaces FFT frequency analysis with actual entailment detection
"""
from transformers import pipeline
import numpy as np

class ReliabilityScorer:
    """
    Scores how much evidence supports a claim using NLI entailment.
    Works for Arabic, Tamil, and English (your three languages).
    """

    def __init__(self):
        # mDeBERTa works for Arabic, Tamil, English — unlike bart-large-mnli which is English only
        self.nli = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=-1  # CPU, change to 0 if you have GPU
        )

    def compute_reliability(self, claim: str, evidence_text: str) -> float:
        """
        Returns probability that evidence supports the claim (0 to 1)

        Args:
            claim: The claim being verified
            evidence_text: A single piece of retrieved evidence

        Returns:
            Entailment probability as reliability score
        """
        result = self.nli(
            evidence_text[:512],  # truncate to avoid token limit
            candidate_labels=["supported", "refuted"],
            hypothesis_template="The following claim is {}: " + claim
        )

        supported_idx = result['labels'].index('supported')
        return float(result['scores'][supported_idx])

    def batch_compute_reliability(self, claim: str, evidence_texts: list) -> np.ndarray:
        """
        Score multiple evidence pieces against one claim

        Args:
            claim: The claim being verified
            evidence_texts: List of evidence strings

        Returns:
            Array of reliability scores
        """
        scores = []
        for text in evidence_texts:
            score = self.compute_reliability(claim, text)
            scores.append(score)
        return np.array(scores)

    def average_reliability(self, claim: str, evidence_texts: list) -> float:
        """
        Average NLI score across all evidence — use this as nli_score in EvidenceGate

        Args:
            claim: The claim being verified
            evidence_texts: List of evidence strings

        Returns:
            Mean reliability score (0 to 1)
        """
        scores = self.batch_compute_reliability(claim, evidence_texts)
        return float(np.mean(scores)) if len(scores) > 0 else 0.0
