# φ3.5 α-gate Mathematical Design Plan

## 1. Parameter Budget Formalization

### 1.1 Symbols and Assumptions
- Base model parameter count: \(P_{\mathrm{base}} \approx 3.8\times 10^9\)
- Transformer specs: depth \(L\), hidden size \(d_{\mathrm{model}}\), attention heads per layer \(M\)
- α-gate inserted on a contiguous middle band of \(L_{\mathrm{mid}}\) layers
- Each α-gate layer contains:
  1. Low-rank projection \(P: \mathbb{R}^{d_{\mathrm{model}}} \rightarrow \mathbb{R}^{d_\alpha}\)
  2. SO(8) rotation parameterized by skew-symmetric matrix \(A_{\ell}\in\mathfrak{so}(8)\)
  3. Scalar mixing coefficient \(\alpha_{\ell}\)

### 1.2 Parameter Count per Layer
Projection parameters per layer: \(d_{\mathrm{model}} d_\alpha\)
Reconstruction (using \(P^T\)): another \(d_{\mathrm{model}} d_\alpha\)
SO(8) rotation degrees of freedom: 28 per skew matrix. Let sharing factor \(S\) denote the number of independent rotations (\(S=1\) for per-layer, \(S=M\) for per-head).
Scalar \(\alpha_{\ell}\): contributes \(S\) parameters if distinct per shared unit.
Therefore:
\[
p_{\alpha}^{(\mathrm{layer})} = 2 d_{\mathrm{model}} d_\alpha + 29 S
\]

### 1.3 Total α Parameters and Overhead Ratio
\[
P_{\alpha} = L_{\mathrm{mid}} \cdot p_{\alpha}^{(\mathrm{layer})}
= L_{\mathrm{mid}} (2 d_{\mathrm{model}} d_\alpha + 29 S)
\]
\[
\rho = \frac{P_{\alpha}}{P_{\mathrm{base}}}
= \frac{L_{\mathrm{mid}} (2 d_{\mathrm{model}} d_\alpha + 29 S)}{P_{\mathrm{base}}}
\]

Given an upper bound \(\rho_{\max}\), solve for feasible \((L_{\mathrm{mid}}, d_\alpha)\):
\[
d_\alpha \le \frac{1}{2 d_{\mathrm{model}}}\left( \frac{\rho_{\max} P_{\mathrm{base}}}{L_{\mathrm{mid}}} - 29 S \right)
\]
This expression guides selection of middle-band width and adapter rank under per-layer (\(S=1\)) or per-head (\(S=M\)) sharing.

## 2. Objective for α-Gate Optimization

### 2.1 Total Loss
\[
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{task}} + \lambda_{\alpha} \mathcal{R}_{\alpha} + \lambda_{\mathrm{SO8}} \mathcal{R}_{\mathrm{SO8}}
\]
- \(\mathcal{L}_{\mathrm{task}}\): supervised objective on coding/reasoning tasks
- \(\mathcal{R}_{\alpha}\): α-gate regularization
- \(\mathcal{R}_{\mathrm{SO8}} = \sum_{\ell} \| R_{\ell}^T R_{\ell} - I \|_F^2\), enforcing near-orthogonality (\(R_{\ell} = \exp(A_{\ell})\))

### 2.2 α Regularization Forms
**Shared α (single scalar):**
\[
\mathcal{R}_{\alpha}^{\mathrm{shared}} = (\alpha - \alpha^\star)^2 + \beta \big( \log(\alpha+\varepsilon)^2 + \log(1-\alpha+\varepsilon)^2 \big)
\]
Ensures attraction to target \(\alpha^\star\) and avoids collapse near 0 or 1.

**Layerwise α:**
\[
\mathcal{R}_{\alpha}^{\mathrm{layer}} = \sum_{\ell\in\mathrm{mid}} \left[(\alpha_{\ell}-\alpha^\star)^2 + \beta \big( \log(
alpha_{\ell}+\varepsilon)^2 + \log(1-\alpha_{\ell}+\varepsilon)^2 \big) \right]
+ \gamma \sum_{\ell\in\mathrm{mid}-1} (\alpha_{\ell+1} - \alpha_{\ell})^2
\]
Includes barrier terms and depth-smoothness regularizer.

**φ⁻² Band Constraint:**
\[
\mathcal{R}_{\alpha}^{(\phi)} = \sum_{\ell\in\mathrm{mid}} (\alpha_{\ell} - \phi^{-2})^2, \quad \phi^{-2} \approx 0.382
\]
Rationale: φ captures self-similarity and balanced scaling. Alternatives: (i) depth-dependent target \(\alpha_{\ell}^\star = \phi^{-2} + \kappa \frac{\ell - L_{\mathrm{mid}}/2}{L_{\mathrm{mid}}}\); (ii) parameter ratio-based target \(\alpha^\star = \sigma(c\rho)\).

## 3. Scaling of α Parameters vs Base Model

### 3.1 Why Sub-linear Growth
- Large frozen backbones already have high capacity; linear growth increases VC dimension and Rademacher complexity without proportional data, risking overfitting
- Sub-linear or logarithmic growth keeps adapters focused on local corrections (residual learning) and maintains stability
- Practical compute on RTX 3060 necessitates tight control of additional parameters, motivating \(P_{\alpha} \in O(P_{\mathrm{base}}^{1/2})\) or \(O(\log P_{\mathrm{base}})\)

### 3.2 Concrete Budgets for phi-3.5 (~3.8B params)
| Mode | \(\rho\) | \(P_{\alpha}\) | Example (\(L_{\mathrm{mid}}, d_\alpha\), per-layer SO(8), \(d_{\mathrm{model}}=2048\)) |
| --- | --- | --- | --- |
| Conservative | 0.001 | ≈3.8M | \(L_{\mathrm{mid}}=8\), \(d_\alpha \approx 116\) |
| Moderate | 0.003 | ≈11.4M | \(L_{\mathrm{mid}}=12\), \(d_\alpha \approx 230\) |
| Aggressive | 0.01 | ≈38M | \(L_{\mathrm{mid}}=16\), \(d_\alpha \approx 580\) |
These configurations stay within ≤10 GB VRAM during fine-tuning when paired with gradient accumulation and mixed precision.

## 4. Training & Annealing Schedule for α

### 4.1 Sigmoid-style Schedule
- **Warmup** (0 ≤ t ≤ T_w): constrain α near 0
  \[ \alpha(t) = \alpha_{\min} + (\alpha_0 - \alpha_{\min}) \cdot \frac{t}{T_w} \]
- **Anneal** (T_w < t ≤ T_a): logistic towards \(\alpha_{\text{target}}\)
  \[ \alpha(t) = \alpha_{\text{target}} \cdot \frac{1}{1 + e^{-k(t - T_a/2)}} \]
- **Cooldown/Fine-tune** (t > T_a): small oscillation for stability
  \[ \alpha(t) = \alpha_{\text{target}} + \delta \cos\Big( \pi \frac{t - T_a}{T_c} \Big) \]
Parameters: \(\alpha_{\min}\approx 0.0\), \(\alpha_0 \approx 0.01\), \(\alpha_{\text{target}}=\phi^{-2}\).

### 4.2 Monitoring Metrics
- Mean α: \(\bar{\alpha}(t) = \frac{1}{L_{\mathrm{mid}}} \sum_{\ell} \alpha_{\ell}(t)\)
- Variance: \(\mathrm{Var}_{\alpha}(t) = \frac{1}{L_{\mathrm{mid}}} \sum_{\ell} (\alpha_{\ell} - \bar{\alpha})^2\)
- Mass-gap proxy: estimate from smallest singular values of mid-layer hidden states

Pseudo-logging:
```python
def log_alpha_metrics(alpha_values, hidden_states):
    alpha_tensor = torch.stack(list(alpha_values.values()))
    alpha_mean = alpha_tensor.mean().item()
    alpha_var = alpha_tensor.var(unbiased=False).item()

    spectra = []
    for h in hidden_states[mid_start:mid_end]:
        svals = torch.linalg.svdvals(h.detach())
        spectra.append(svals[-1])  # smallest singular value
    mass_gap = torch.stack(spectra).mean().item()

    logger.info({
        "alpha_mean": alpha_mean,
        "alpha_var": alpha_var,
        "mass_gap": mass_gap,
    })
```

## 5. Bayesian / Black-box Optimization Outline (Optional)
- **Search space** (Optuna):
  - \(\alpha^\star \in [0.25, 0.55]\)
  - \(\rho \in [5\times 10^{-4}, 10^{-2}]\)
  - \(L_{\mathrm{mid}} \in \{6,8,10,12,16\}\)
- **Derived variable**: \(d_\alpha\) from budget constraint
- **Objective**: validation loss + code-quality metrics (e.g., pass@k, bug-fix accuracy) + penalty on \(\rho\)
- **Stopping**: max trials, convergence of objective, or VRAM constraint violation (pruned)

## 6. High-level Implementation Sketch

```python
# Freeze phi-3.5 backbone
tokenizer, base_model = load_phi35(frozen=True)
mid_layers = range(mid_start, mid_end)

# Insert residual α-gate adapters
adapters = {}
total_params = 0
for idx in mid_layers:
    adapter = SO8TAdapter(d_model=d_model,
                          d_alpha=d_alpha,
                          shared_rotation=(sharing=='layer'),
                          n_heads=M)
    adapters[idx] = adapter
    total_params += adapter.count_params()

rho_actual = total_params / P_base
assert rho_actual <= rho_budget

# Forward with adapters
outputs = base_model(input_ids, attention_mask, output_hidden_states=True)
hidden_states = list(outputs.hidden_states)
alpha_values = schedule(step)

for layer_idx, adapter in adapters.items():
    hidden_states[layer_idx] = adapter(hidden_states[layer_idx], alpha=alpha_values[layer_idx])

last_hidden = hidden_states[-1]
logits = base_model.lm_head(last_hidden)

# Loss components
task_loss = cross_entropy(logits[:, :-1], labels[:, 1:])
alpha_reg = sum(alpha_regularizer(alpha_values[idx], target=alpha_target)
                for idx in mid_layers)
so8_reg = sum(adapter.orthogonality_penalty() for adapter in adapters.values())

loss = task_loss + lambda_alpha * alpha_reg + lambda_so8 * so8_reg

# Training loop (AMP + grad accumulation)
optimizer = AdamW(adapters_parameters, lr)
scaler = GradScaler()

for step, batch in enumerate(dataloader):
    with autocast():
        loss = compute_loss(batch, step)
    scaler.scale(loss).backward()

    if (step + 1) % grad_accum == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    log_alpha_metrics(alpha_values, hidden_states)

# Export: absorb adapters into GGUF-compatible weights if needed
```

Key considerations:
- Keep adapters lightweight enough for ≤10 GB VRAM (use gradient accumulation/mixed precision)
- Ensure α schedule and regularizers drive values toward φ⁻² band
- Maintain ability to export combined weights into GGUF format for deployment
- Budget check enforced before training
```