# LLM Benchmark A/B Test Report

**Generated:** 2025-11-23 12:38:00

## Models Tested

- model-a:q8_0
- agiasi-phi35-golden-sigmoid:q8_0

## Overall Performance

| model                            |   total_questions |   overall_accuracy |   avg_tokens_per_sec |   avg_wall_time |
|:---------------------------------|------------------:|-------------------:|---------------------:|----------------:|
| agiasi-phi35-golden-sigmoid:q8_0 |                10 |                0.3 |              66.4672 |          5.8133 |
| model-a:q8_0                     |                10 |                0.5 |              64.8678 |          9.3785 |

## Performance by Task

| model                            | task                  |   count |   accuracy |   avg_tokens_per_sec |   avg_wall_time |
|:---------------------------------|:----------------------|--------:|-----------:|---------------------:|----------------:|
| agiasi-phi35-golden-sigmoid:q8_0 | gsm8k                 |       5 |        0.6 |              65.9293 |          4.5889 |
| agiasi-phi35-golden-sigmoid:q8_0 | mmlu_abstract_algebra |       5 |        0   |              67.0051 |          7.0378 |
| model-a:q8_0                     | gsm8k                 |       5 |        1   |              65.6436 |          4.2442 |
| model-a:q8_0                     | mmlu_abstract_algebra |       5 |        0   |              64.092  |         14.5128 |

## Key Findings

### Winner: model-a:q8_0
- Accuracy difference: 20.0%
- Faster model: agiasi-phi35-golden-sigmoid:q8_0 (1.6 tokens/sec faster)

## Visualizations

![Benchmark Comparison](benchmark_comparison.png)

## Raw Results

### model-a:q8_0
- Correct answers: 5/10
- Accuracy: 50.0%

### agiasi-phi35-golden-sigmoid:q8_0
- Correct answers: 3/10
- Accuracy: 30.0%

