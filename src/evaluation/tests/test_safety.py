import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import torch
    from src.core.so8t_core.self_verification import SelfVerifier
    from src.core.so8t_core.triality_heads import TrialityHead
    _IMPORTS_AVAILABLE = True
except ImportError:
    _IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not _IMPORTS_AVAILABLE, reason="so8t_core modules not available")
def test_triality_head_probs_sum_to_one():
    head = TrialityHead(hidden_size=128)
    hidden = torch.randn(1, 4, 128)
    output = head(hidden)
    assert torch.isclose(output.probabilities.sum(), torch.tensor(1.0), atol=1e-5)


@pytest.mark.skipif(not _IMPORTS_AVAILABLE, reason="so8t_core modules not available")
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
