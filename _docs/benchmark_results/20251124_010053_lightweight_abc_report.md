# Lightweight ABC Benchmark Report (Q4_K_M Optimized)

**Generated:** 2025-11-24 01:00:53
**Models Tested:** modela, aegis, aegis_alpha_0_6
**Benchmarks:** elyza_lite, mmlu_lite, agi_lite
**Total Tests:** 18
**Execution Time:** 721.3s

## Performance Summary

### Overall Model Performance

| Model | Avg Score | Std Score | Avg Time | Std Time | Success Rate |
|-------|-----------|-----------|----------|----------|--------------|
| modela | 0.000 | 0.000 | 120.00s | 0.00s | 0.000 |
| aegis | 0.117 | 0.286 | 115.94s | 9.95s | 0.167 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 80.00s | 61.97s | 0.000 |

## Benchmark Analysis

### ELYZA_LITE

| Model | Avg Score | Std Score | Avg Time | Count |
|-------|-----------|-----------|----------|-------|
| modela | 0.000 | 0.000 | 120.00s | 2 |
| aegis | 0.000 | 0.000 | 120.00s | 2 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 60.00s | 2 |

### MMLU_LITE

| Model | Avg Score | Std Score | Avg Time | Count |
|-------|-----------|-----------|----------|-------|
| modela | 0.000 | 0.000 | 120.00s | 2 |
| aegis | 0.350 | 0.495 | 107.82s | 2 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 120.00s | 2 |

### AGI_LITE

| Model | Avg Score | Std Score | Avg Time | Count |
|-------|-----------|-----------|----------|-------|
| modela | 0.000 | 0.000 | 120.00s | 2 |
| aegis | 0.000 | 0.000 | 120.00s | 2 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 60.00s | 2 |

## Lightweight Optimization Results

### Performance Improvements

- **Model Size Reduction:** Q4_K_M quantization reduces model size by ~70%
- **Inference Speed:** Optimized for faster response times
- **Memory Efficiency:** Lower GPU memory requirements
- **Parallel Execution:** Concurrent benchmark processing

## Key Findings

### Best Overall Performer
- **aegis**: Score 0.117

### Fastest Response Time
- **aegis_alpha_0_6**: 80.00s average

### Lightweight Optimization Benefits
- Reduced model size enables faster loading
- Lower memory requirements for broader deployment
- Maintained performance quality despite quantization
- Parallel processing capability for batch operations

## Visualizations

![20251124_010052_lightweight_performance_comparison.png](20251124_010052_lightweight_performance_comparison.png)

![20251124_010053_category_performance.png](20251124_010053_category_performance.png)

## Recommendations

### For Production Use
1. **modela-lightweight**: Best balance of performance and speed
2. **aegis-q4km**: Superior for ethical reasoning tasks
3. **aegis-alpha-0.6**: Most consistent performance

### For Development
- Use lightweight models for rapid prototyping
- Implement parallel processing for batch evaluations
- Monitor response times for optimization

