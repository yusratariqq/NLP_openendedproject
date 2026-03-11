"""
Evidence support ratio calculation
"""
import numpy as np
from typing import Dict

class SupportRatioCalculator:
    """Calculate evidence support ratio β"""

    def __init__(self, support_threshold: float = 0.85):
        """
        Args:
            support_threshold: Threshold for considering evidence as supporting
        """
        self.support_threshold = support_threshold

    def compute_ratio(self, fidelities: np.ndarray) -> float:
        """
        Compute β = (# supporting evidence) / total

        Args:
            fidelities: Array of fidelity scores

        Returns:
            Support ratio β (0 to 1)
        """
        if len(fidelities) == 0:
            return 0.0

        supporting = np.sum(fidelities >= self.support_threshold)
        total = len(fidelities)

        beta = supporting / total

        return float(beta)

    def get_support_details(self, fidelities: np.ndarray) -> Dict:
        """
        Return detailed breakdown of evidence support

        Args:
            fidelities: Array of fidelity scores

        Returns:
            Dictionary with support statistics
        """
        if len(fidelities) == 0:
            return {
                'supporting': 0,
                'contradicting': 0,
                'neutral': 0,
                'total': 0,
                'ratio': 0.0
            }

        supporting = np.sum(fidelities >= self.support_threshold)
        contradicting = np.sum(fidelities < 0.70)
        neutral = np.sum((fidelities >= 0.70) & (fidelities < self.support_threshold))

        return {
            'supporting': int(supporting),
            'contradicting': int(contradicting),
            'neutral': int(neutral),
            'total': len(fidelities),
            'ratio': float(supporting / len(fidelities))
        }

    def classify_evidence(self, fidelity: float) -> str:
        """
        Classify single evidence as supporting/neutral/contradicting

        Args:
            fidelity: Fidelity score

        Returns:
            Classification string
        """
        if fidelity >= self.support_threshold:
            return 'SUPPORTING'
        elif fidelity < 0.3:
            return 'CONTRADICTING'
        else:
            return 'NEUTRAL'
