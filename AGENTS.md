# AGENTS.md - SO8T Multimodal MoE Training Pipeline

## Overview

SO8T (Aegis-VSSI) autonomous AI research pipeline with **multimodal MoE architecture** and SO8 group triality-based routing. Implements ShinkaEvolve and AIScentist-inspired frozen parameter evolution using Ebbinghaus forgetting curve dynamics.

## Hardware Configuration

- **GPU**: RTX 3060+ (12GB VRAM)
- **CPU**: Ryzen 5600 12-core
- **RAM**: 32GB
- **Storage**: D:\webdataset (50GB+)

## Pipeline Architecture

### Phase 1: Data Collection & Processing
```
Antigravity Browser (Playwright) → Data Processing → SFT+GRPO Training
```

### Phase 2: Advanced Training (SFT+GRPO+mHC+GRAPE+imatrix+PET+Unsloth BF16)
```
SFT → mHC Manifold Alignment → GRPO → GRAPE Position Encoding → imatrix Quantization → PET Regularization
```

### Phase 3: SO8T Multimodal MoE Evolution
```
ShinkaEvolve Frozen Parameter Evolution → Ebbinghaus Forgetting Curve → SO8 Triality Routing (Multimodal)
```

### Phase 4: C/D MoE Testing & HF Upload
```
Model C MoE Conversion → C/D Comparative Test → SafeTensors Upload → BF16 GGUF Upload
```

## Multimodal MoE Architecture

```
Input (Text + Images)
        ↓
┌───────────────────────┐
│   Vision Encoder       │  CLIP ViT → Projection Layer
│   (CLIP ViT-base)     │
└───────────────────────┘
        ↓
┌───────────────────────┐
│   SO8 Triality Router  │  Vector/Spinor indistinguishability
│   (4 Experts)          │  Routing weights: softmax(gate)
└───────────────────────┘
        ↓
┌───────────────────────┐
│   Expert Layers        │  [Expert A, Expert B, Expert C, Expert D]
│   (MoE Selection)      │  Top-K gating (k=2)
└───────────────────────┘
        ↓
Output (Text Generation)
```

## Implementation Details

### SO8 Group Triality for Multimodal MoE Routing

The SO8 group exhibits triality - an automorphism that permutes vectors, spinors, and conjugate spinors. This property enables indistinguishable routing between:
- **Vector representation**: Standard token embeddings
- **Positive spinor**: Image features (vision pathway)
- **Negative spinor**: Hidden states (safety pathway)

```python
# src/core/models/so8t_moe_router.py
class SO8TrialityRouter:
    def __init__(self, num_experts: int, hidden_dim: int):
        self.vector_proj = nn.Linear(hidden_dim, hidden_dim)
        self.spinor_pos = nn.Linear(hidden_dim, hidden_dim)  # Image features
        self.spinor_neg = nn.Linear(hidden_dim, hidden_dim)  # Safety pathway
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_experts),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Triality-based routing: vectors/spinors indistinguishability
        vector_state = self.vector_proj(x)
        spinor_pos = self.spinor_pos(x)
        spinor_neg = self.spinor_neg(x)
        triality_states = torch.stack([vector_state, spinor_pos, spinor_neg], dim=2)
        routing_weights = F.softmax(self.gate(triality_states.mean(dim=(1, 2))), dim=-1)
        return expert_indices, routing_weights
```

### Vision Encoder

```python
# src/training/so8t_multimodal_moe_pipeline.py
class VisionEncoder(nn.Module):
    def __init__(self, config: SO8MultimodalMoEConfig):
        self.encoder = CLIPVisionModel.from_pretrained(config.vision_model_name)
        self.projection = nn.Linear(vision_hidden_size, config.hidden_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=images)
        return self.projection(outputs.last_hidden_state)
```

### Ebbinghaus Forgetting Curve Integration

Dynamically evolve frozen parameters based on token usage frequency:

```python
# src/training/evolution/ebbinghaus_forgetting.py
class EbbinghausForgettingCurve:
    def __init__(self, decay_rate: float = 0.1, retention_threshold: float = 0.3):
        self.token_states: Dict[int, Dict[str, float]] = {}
        
    def update(self, token_ids: List[int], is_reinforced: bool = False) -> None:
        for token_id in token_ids:
            if token_id not in self.token_states:
                self.token_states[token_id] = {'retention': 1.0, 'usage_count': 0}
            if is_reinforced:
                self.token_states[token_id]['retention'] = min(1.0, 
                    self.token_states[token_id]['retention'] + 0.1)
            else:
                self.token_states[token_id]['retention'] *= (1 - self.decay_rate)
                
    def get_frozen_param_multiplier(self, param_name: str) -> float:
        return 0.5 * self.manifold_scaling_factor
```

### ShinkaEvolve-Inspired Frozen Parameter Evolution

```python
# src/training/evolution/shinka_evolve.py
class ShinkaEvolveOptimizer:
    def __init__(self, model: nn.Module, ebbinghaus: EbbinghausForgettingCurve):
        self.model = model
        self.ebbinghaus = ebbinghaus
        self.frozen_params: Set[str] = set()
        
    def evolve_frozen_parameters(self, step: int) -> Dict:
        for name, param in self.model.named_parameters():
            if name in self.frozen_params:
                retention = self.ebbinghaus.get_frozen_param_multiplier(name)
                if retention < self.retention_threshold:
                    param.data += torch.randn_like(param) * 0.01 * (1 - retention)
        return {'step': step, 'active_frozen': len(self.frozen_params)}
```

### Rolling Checkpoint System (5-minute intervals, 3 slots)

```python
# src/utils/checkpoint_manager.py
class RollingCheckpointManager:
    def __init__(self, checkpoint_dir: str, interval_seconds: int = 300, max_slots: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval_seconds = interval_seconds
        self.max_slots = max_slots
        
    def save_checkpoint(self, model: nn.Module, step: int, is_emergency: bool = False) -> None:
        slot_path = self.checkpoint_dir / f"checkpoint_slot_{self.current_slot}.pt"
        torch.save({'model_state': model.state_dict(), 'step': step}, slot_path)
        self.current_slot = (self.current_slot + 1) % self.max_slots
```

### Progress Tracking (tqdm + logging)

```python
# src/utils/progress_tracker.py
class TrainingProgressTracker:
    def __init__(self, total_steps: int, desc: str = "Training"):
        self.pbar = tqdm(total=total_steps, desc=desc)
        
    def update(self, step: int, metrics: Dict[str, float]) -> None:
        self.pbar.update(1)
        self.pbar.set_postfix(metrics)
```

## Model Configuration

### Models A, B, C, D

| Model | Location | Purpose | GGUF |
|-------|----------|---------|------|
| A | H:\from_D\SO8T_models\model_a | Baseline (Phi-3.5-mini-instinct) | ✅ model_a.bf16.gguf |
| B | H:\from_D\SO8T_models\model_b | Comparison (Borea-Phi-3.5-mini-Jp) | ✅ model_b.bf16.gguf |
| C | H:\from_D\SO8T_models\model_c | MoE Conversion Target | ✅ model_c.bf16.gguf |
| D | H:\from_D\SO8T_models\model_d | SO8T Multimodal MoE Gateway | pending |

### SO8T Multimodal MoE Architecture

```
Input (Text + Images)
    ↓
Vision Encoder (CLIP ViT)
    ↓
Image Token Projection (768 → 3072)
    ↓
SO8 Triality Router (4 experts)
    ↓
[Expert A | Expert B | Expert C | Expert D]
    ↓
Top-K Gating (k=2)
    ↓
Output
```

## Training Command

```powershell
# Multimodal MoE Training
py -3 src/training/so8t_multimodal_moe_pipeline.py \
  --model-name "microsoft/Phi-3.5-mini-instruct" \
  --vision-model "openai/clip-vit-base-patch32" \
  --output-dir "D:\webdataset\models\so8t_multimodal_moe" \
  --num-experts 4 \
  --batch-size 4 \
  --epochs 3 \
  --max-steps 10000 \
  --multimodal

# Full Pipeline (requires admin)
.\scripts\pipeline\run_so8t_moe_pipeline.ps1

# Manual Resume from Checkpoint
python scripts/pipeline/auto_resume_aegis.py

# Quick Test
py -3 simple_rlpo_test.py
```

## Output Paths

- **Checkpoints**: D:\webdataset\checkpoints\training\
- **GGUF Models**: D:\webdataset\gguf_models\
- **Final Models**: D:\webdataset\models\so8t_multimodal_moe\
- **HF Upload Package**: hf_upload_package\
- **Logs**: logs\so8t_multimodal_moe.log

## Code Style

- PEP 8 compliant
- Type hints required for all public functions
- Google-style docstrings
- Black formatting
- isort imports
- mypy type checking
- No emojis in code/comments (use [OK], [NG], etc.)
- UTF-8 encoding

## References

1. **ShinkaEvolve**: Evolutionary optimization of frozen parameters
2. **AIScentist**: Manifold-based scaling and knowledge retention
3. **SO(8) NKAT Theory**: Non-commutative kernel adaptation for triality
4. **DeepSeek-V3 GRPO**: Group Relative Policy Optimization
5. **Ebbinghaus Forgetting Curve**: Retention dynamics for token embeddings
6. **Unsloth 4-bit QLoRA**: Memory-efficient fine-tuning for RTX 3060
7. **CLIP**: Vision-Language Pretraining
8. **Multimodal MoE**: mixture-of-experts for vision-language models
