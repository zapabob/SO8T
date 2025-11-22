# NKAT-SO8T Dataset Quality Control Report
Generated: 2025-11-22 15:26:44

## Dataset Overview
- **Total Samples**: 120
- **Cleansing Status**: ✅ Completed

## Quality Metrics

### Text Statistics
- **Mean Length**: 33.6 chars
- **Median Length**: 32.0 chars
- **Length Range**: 21 - 90 chars

### Token Statistics
- **Mean Tokens**: 40.5
- **Median Tokens**: 39.0
- **Token Range**: 24 - 127

### NKAT-SO8T Specific Metrics
- **Reasoning Depth**: 0.152
- **Mathematical Content**: 0.004
- **Logical Structure**: 0.000

## Statistical Tests
### Normality Tests (Shapiro-Wilk)
- **Text Length**: p = 3.1999197498111576e-20
- **Token Length**: p = 7.054189667273119e-20
- **Complexity**: p = 3.219626713288106e-13

### Outlier Analysis
- **Outliers Detected**: 10 (8.3%)

## Quality Control Status
- ✅ Basic quality filters applied
- ✅ Statistical deduplication completed
- ✅ NKAT-SO8T specific filtering applied
- ✅ Category balancing completed
- ✅ Outlier removal performed

## Recommendations
- Dataset quality: Needs Improvement
- Ready for NKAT-SO8T training: ❌ No - Insufficient samples
