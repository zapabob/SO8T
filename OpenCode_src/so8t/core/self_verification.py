"""
Self-verification utilities combining multi-pass reasoning and rule checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol

from torch import Tensor


class SupportsLog(Protocol):
    def log(self, message: str) -> None:
        ...


@dataclass
class VerificationResult:
    choice: str
    score: float
    reasoning: str
    votes: List[float]


class SelfVerifier:
    """
    Aggregates multiple reasoning passes and scores them for final decision making.
    """

    def __init__(
        self,
        consistency_weight: float = 0.4,
        numeric_weight: float = 0.3,
        policy_weight: float = 0.3,
        logger: Optional[SupportsLog] = None,
    ) -> None:
        self.consistency_weight = consistency_weight
        self.numeric_weight = numeric_weight
        self.policy_weight = policy_weight
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    def score_pass(
        self,
        reasoning: str,
        logits: Tensor,
        compliance_score: float,
    ) -> float:
        norm_logits = logits.softmax(dim=-1)
        entropy = -(norm_logits * norm_logits.log()).sum().item()
        consistency = 1.0 - min(entropy / 5.0, 1.0)
        numeric = float(norm_logits.max().item())
        score = (
            consistency * self.consistency_weight
            + numeric * self.numeric_weight
            + compliance_score * self.policy_weight
        )
        self._log(f"pass score={score:.3f} consistency={consistency:.3f} numeric={numeric:.3f}")
        return score

    def verify(
        self,
        reasoning_passes: Iterable[str],
        logits: Iterable[Tensor],
        compliance_scores: Iterable[float],
        labels: Iterable[str],
    ) -> VerificationResult:
        reasoning_list = list(reasoning_passes)
        logits_list = list(logits)
        compliance_list = list(compliance_scores)
        labels_list = list(labels)

        best_index = -1
        best_score = float("-inf")
        best_reasoning = ""
        votes: List[float] = []

        for idx, (reasoning, logit, compliance, label) in enumerate(
            zip(reasoning_list, logits_list, compliance_list, labels_list)
        ):
            score = self.score_pass(reasoning, logit, compliance)
            votes.append(score)
            self._log(f"label={label} vote={score:.3f}")
            if score > best_score:
                best_index = idx
                best_score = score
                best_reasoning = reasoning

        final_label = labels_list[best_index] if best_index >= 0 else "ESCALATE"
        return VerificationResult(choice=final_label, score=best_score, reasoning=best_reasoning, votes=votes)
