"""
Evidence confidence gate - decides if claim has enough support to answer
"""
from typing import Dict

class EvidenceGate:
    """
    Controls whether the system should answer based on three signals:
    - nli_score:   Does evidence actually entail the claim? (NLI score)
    - similarity:  Are retrieved docs semantically close? (fidelity)
    - support_ratio: What fraction of evidence supports the claim? (beta)
    """

    def __init__(self, nli_min: float = 0.55, similarity_min: float = 0.80, support_min: float = 0.40):
        self.nli_min = nli_min
        self.similarity_min = similarity_min
        self.support_min = support_min

    def should_answer(self, nli_score: float, similarity: float, support_ratio: float) -> bool:
        return (
            nli_score >= self.nli_min and
            similarity >= self.similarity_min and
            support_ratio >= self.support_min
        )

    def get_verdict(self, nli_score: float, similarity: float, support_ratio: float) -> Dict:
        nli_ok = nli_score >= self.nli_min
        sim_ok = similarity >= self.similarity_min
        sup_ok = support_ratio >= self.support_min

        gates_passed = sum([nli_ok, sim_ok, sup_ok])

        # Meaningful verdict based on which gates failed
        if gates_passed == 3:
            verdict = "SUPPORTED"
        elif gates_passed == 0:
            verdict = "REFUTED"
        elif not nli_ok:
            verdict = "INSUFFICIENT_EVIDENCE"   # docs retrieved but don't entail claim
        elif not sup_ok:
            verdict = "CONFLICTING_EVIDENCE"    # some support but not enough
        else:
            verdict = "UNVERIFIABLE"            # borderline case

        return {
            'verdict': verdict,
            'should_answer': gates_passed == 3,
            'gates_passed': gates_passed,
            'details': {
                'nli_score':     {'value': nli_score,     'threshold': self.nli_min,        'passed': nli_ok},
                'similarity':    {'value': similarity,    'threshold': self.similarity_min,  'passed': sim_ok},
                'support_ratio': {'value': support_ratio, 'threshold': self.support_min,     'passed': sup_ok},
            }
        }

    def update_thresholds(self, nli_min=None, similarity_min=None, support_min=None):
        if nli_min is not None:
            self.nli_min = nli_min
        if similarity_min is not None:
            self.similarity_min = similarity_min
        if support_min is not None:
            self.support_min = support_min
