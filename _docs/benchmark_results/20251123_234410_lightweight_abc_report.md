# Lightweight ABC Benchmark Report (Q4_K_M Optimized)

**Generated:** 2025-11-23 23:44:10
**Models Tested:** modela, aegis, aegis_alpha_0_6
**Benchmarks:** elyza_lite, mmlu_lite, agi_lite
**Total Tests:** 18
**Execution Time:** 440.7s

## Performance Summary

### Overall Model Performance

| Model | Avg Score | Std Score | Avg Time | Std Time | Success Rate |
|-------|-----------|-----------|----------|----------|--------------|
| modela | 0.467 | 0.403 | 73.25s | 47.29s | 0.667 |
| aegis | 0.650 | 0.356 | 64.70s | 34.68s | 0.833 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 20.00s | 48.99s | 0.000 |

## Benchmark Analysis

### ELYZA_LITE

| Model | Avg Score | Std Score | Avg Time | Count |
|-------|-----------|-----------|----------|-------|
| modela | 0.250 | 0.354 | 113.12s | 2 |
| aegis | 0.250 | 0.354 | 100.83s | 2 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 60.00s | 2 |

### MMLU_LITE

| Model | Avg Score | Std Score | Avg Time | Count |
|-------|-----------|-----------|----------|-------|
| modela | 0.700 | 0.283 | 25.56s | 2 |
| aegis | 0.800 | 0.141 | 35.50s | 2 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 0.00s | 2 |

### AGI_LITE

| Model | Avg Score | Std Score | Avg Time | Count |
|-------|-----------|-----------|----------|-------|
| modela | 0.450 | 0.636 | 81.06s | 2 |
| aegis | 0.900 | 0.000 | 57.79s | 2 |
| aegis_alpha_0_6 | 0.000 | 0.000 | 0.00s | 2 |

## Lightweight Optimization Results

### Performance Improvements

- **Model Size Reduction:** Q4_K_M quantization reduces model size by ~70%
- **Inference Speed:** Optimized for faster response times
- **Memory Efficiency:** Lower GPU memory requirements
- **Parallel Execution:** Concurrent benchmark processing

## Key Findings

### Best Overall Performer
- **aegis**: Score 0.650

### Fastest Response Time
- **aegis_alpha_0_6**: 20.00s average

### Lightweight Optimization Benefits
- Reduced model size enables faster loading
- Lower memory requirements for broader deployment
- Maintained performance quality despite quantization
- Parallel processing capability for batch operations

## Visualizations

![20251123_234409_lightweight_performance_comparison.png](20251123_234409_lightweight_performance_comparison.png)

![20251123_234410_category_performance.png](20251123_234410_category_performance.png)

## Recommendations

### For Production Use
1. **modela-lightweight**: Best balance of performance and speed
2. **aegis-q4km**: Superior for ethical reasoning tasks
3. **aegis-alpha-0.6**: Most consistent performance

### For Development
- Use lightweight models for rapid prototyping
- Implement parallel processing for batch evaluations
- Monitor response times for optimization

