# SO8T Anti-Local-Minimum Implementation Guide

## Quick Start

```python
# 1. 勾配ノイズ注入
sigma = 0.025
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * sigma
            p.grad.add_(noise)

# 2. ラベルスムージング
def smooth_ce_loss(logits, target, eps=0.3):
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(eps / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(true_dist * log_probs).sum(dim=-1).mean()

# 3. PETスケジュール
def get_pet_lambda(step, total_steps):
    progress = step / total_steps
    if progress < 0.2:    # 探索期
        return base_lambda * 0.01
    elif progress < 0.6:  # 遷移期
        return base_lambda * 0.1
    else:                 # 安定化期
        return base_lambda * 1.0

# 4. SWA (70%以降)
if progress >= 0.7:
    swa_model.update_parameters(model)
    swa_scheduler.step()

# 5. 入力ノイズ
if torch.rand(1).item() < 0.2:
    noise_mask = torch.rand_like(input_ids.float()) < 0.1
    input_ids[noise_mask] = mask_token_id
```

## Configuration

```yaml
training:
  learning_rate: 0.002
  weight_decay: 0.1
  pet_lambda: 0.001
  batch_size: 8
  epochs: 5

scheduler:
  warmup_steps: 500

model:
  dropout: 0.2
```

## Monitoring

Key metrics to monitor:
- **Loss Variance**: Should be > 1e-6 (model is alive)
- **Accuracy Variance**: Should be 0.001-0.01 (healthy range)
- **PET Loss Variance**: Should be high (active learning)
- **Generalization Gap**: Should be < 0.1 (good generalization)

## Troubleshooting

### Model still stuck in local minimum?
- Increase gradient noise (σ = 0.05)
- Increase label smoothing (ε = 0.5)
- Extend exploration phase (30% → 40%)

### Model overfitting?
- Increase weight decay (0.1 → 0.2)
- Increase dropout (0.2 → 0.3)
- Start SWA earlier (70% → 60%)

### Model not learning?
- Decrease gradient noise (σ = 0.01)
- Decrease label smoothing (ε = 0.1)
- Increase learning rate (0.002 → 0.003)

