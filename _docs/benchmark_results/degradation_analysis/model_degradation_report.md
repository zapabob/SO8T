# SO8T Model Degradation Analysis Report

**Generated**: 2025-11-25 07:38:06

## Executive Summary

- **Average Degradation Rate**: -10.79%
- **Maximum Degradation Rate**: 100.00%
- **Minimum Degradation Rate**: -993.29%
- **Total Model Pairs Analyzed**: 24

- **Statistically Significant Degradations**: 10/24

## Overall Degradation Metrics

| Baseline Model | SO8T Model | Baseline Mean | SO8T Mean | Absolute Degradation | Degradation Rate (%) | p-value | Significant |
|----------------|------------|---------------|-----------|-------------------|----------------------|---------|-------------|
| model_a | agiasi | 0.027 | 0.049 | -0.022 | -78.57% | 0.1010 | No |
| model_a | aegis | 0.027 | 0.135 | -0.108 | -393.64% | 0.0013 | Yes |
| model_a | aegis_alpha_0_6 | 0.027 | 0.000 | 0.027 | 100.00% | 0.0041 | Yes |
| model_a | aegis-q4km | 0.027 | 0.000 | 0.027 | 100.00% | 0.2485 | No |
| model_a | aegis-alpha-0.6-q4km | 0.027 | 0.000 | 0.027 | 100.00% | 0.2485 | No |
| model_a | AEGIS-phi35-golden-sigmoid | 0.027 | 0.300 | -0.273 | -993.29% | 0.0000 | Yes |
| modela | agiasi | 0.247 | 0.049 | 0.198 | 80.13% | 0.0003 | Yes |
| modela | aegis | 0.247 | 0.135 | 0.111 | 45.07% | 0.0518 | No |
| modela | aegis_alpha_0_6 | 0.247 | 0.000 | 0.247 | 100.00% | 0.0000 | Yes |
| modela | aegis-q4km | 0.247 | 0.000 | 0.247 | 100.00% | 0.0581 | No |
| modela | aegis-alpha-0.6-q4km | 0.247 | 0.000 | 0.247 | 100.00% | 0.0581 | No |
| modela | AEGIS-phi35-golden-sigmoid | 0.247 | 0.300 | -0.053 | -21.66% | 0.6769 | No |
| modela-q4km | agiasi | 0.000 | 0.049 | -0.049 | 0.00% | 0.1004 | No |
| modela-q4km | aegis | 0.000 | 0.135 | -0.135 | 0.00% | 0.2036 | No |
| modela-q4km | aegis_alpha_0_6 | 0.000 | 0.000 | 0.000 | 0.00% | N/A | No |
| modela-q4km | aegis-q4km | 0.000 | 0.000 | 0.000 | 0.00% | N/A | No |
| modela-q4km | aegis-alpha-0.6-q4km | 0.000 | 0.000 | 0.000 | 0.00% | N/A | No |
| modela-q4km | AEGIS-phi35-golden-sigmoid | 0.000 | 0.300 | -0.300 | 0.00% | 0.1246 | No |
| model-a | agiasi | 0.500 | 0.049 | 0.451 | 90.20% | 0.0000 | Yes |
| model-a | aegis | 0.500 | 0.135 | 0.365 | 72.91% | 0.0009 | Yes |
| model-a | aegis_alpha_0_6 | 0.500 | 0.000 | 0.500 | 100.00% | 0.0000 | Yes |
| model-a | aegis-q4km | 0.500 | 0.000 | 0.500 | 100.00% | 0.0252 | Yes |
| model-a | aegis-alpha-0.6-q4km | 0.500 | 0.000 | 0.500 | 100.00% | 0.0252 | Yes |
| model-a | AEGIS-phi35-golden-sigmoid | 0.500 | 0.300 | 0.200 | 40.00% | 0.3880 | No |

## Category-wise Degradation Analysis

| Category | Mean Degradation (%) | Max Degradation (%) | Min Degradation (%) | Count |
|----------|---------------------|---------------------|---------------------|-------|
| agi_abstract_thinking | 0.00 | 0.00 | 0.00 | 1 |
| agi_causal_reasoning | 0.00 | 0.00 | 0.00 | 1 |
| agi_creative_problem_solving | 0.00 | 0.00 | 0.00 | 1 |
| agi_ethical_dilemmas | 0.00 | 0.00 | 0.00 | 1 |
| agi_self_reflection | 0.00 | 0.00 | 0.00 | 1 |
| cyber_defense | 84.48 | 100.00 | 68.97 | 2 |
| defect_analysis | 66.67 | 100.00 | 33.33 | 2 |
| design_verification | 50.00 | 100.00 | 0.00 | 2 |
| general_knowledge | 28.57 | 100.00 | -42.86 | 2 |
| gsm8k | 40.00 | 40.00 | 40.00 | 1 |
| japanese_language | 0.00 | 0.00 | 0.00 | 2 |
| mathematical_reasoning | 28.57 | 100.00 | -42.86 | 2 |
| medical_financial | 0.00 | 0.00 | 0.00 | 2 |
| mission_planning | 50.00 | 100.00 | 0.00 | 2 |
| mmlu_abstract_algebra | 0.00 | 0.00 | 0.00 | 1 |
| power_grid | 75.00 | 100.00 | 50.00 | 2 |
| process_optimization | 50.00 | 100.00 | 0.00 | 2 |
| scientific_knowledge | 0.00 | 0.00 | 0.00 | 2 |
| security_ethics | 0.00 | 0.00 | 0.00 | 2 |
| situational_awareness | 42.86 | 100.00 | -14.29 | 2 |
| structural_analysis | 0.00 | 0.00 | 0.00 | 2 |
| system_diagnostics | 66.67 | 100.00 | 33.33 | 2 |
| threat_detection | -66.67 | 100.00 | -233.33 | 2 |
| trajectory_calculation | 66.67 | 100.00 | 33.33 | 2 |
| transportation | 66.67 | 100.00 | 33.33 | 2 |
| unknown | 38.12 | 100.00 | 0.00 | 8 |

## Key Findings

### Worst Degradation
- **Baseline**: model_a
- **SO8T**: aegis_alpha_0_6
- **Degradation Rate**: 100.00%
- **Absolute Degradation**: 0.027

## Recommendations

1. **Investigate Architecture Changes**: Review SO(8) rotation gate implementation
2. **Check Training Process**: Verify fine-tuning parameters and regularization
3. **Analyze Category-specific Issues**: Focus on categories with highest degradation
4. **Compare Model Sizes**: Check if quantization affects performance
5. **Review Evaluation Metrics**: Ensure fair comparison between models
