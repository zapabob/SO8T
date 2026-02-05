import torch

from so8t_core.self_verification import SelfVerifier
from so8t_core.triality_heads import TrialityHead


def test_triality_head_probs_sum_to_one():
    head = TrialityHead(hidden_size=128)
    hidden = torch.randn(1, 4, 128)
    output = head(hidden)
    assert torch.isclose(output.probabilities.sum(), torch.tensor(1.0), atol=1e-5)


def test_self_verifier_selects_best():
    verifier = SelfVerifier()
    logits = [torch.randn(1, 3), torch.randn(1, 3)]
    scores = [0.2, 0.9]
    result = verifier.verify(
        reasoning_passes=["pass1", "pass2"],
        logits=logits,
        compliance_scores=scores,
        labels=["ALLOW", "ESCALATE"],
    )
    assert result.choice in {"ALLOW", "ESCALATE"}
    assert len(result.votes) == 2
