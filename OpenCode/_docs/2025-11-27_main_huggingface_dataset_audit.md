# HuggingFace Dataset Audit Report

Generated: 2025-11-27 20:22:19

## Summary

- Total datasets: 13
- Existing datasets: 10
- Missing datasets: 3

## Dataset Details

| Dataset | Exists | Samples | NSFW | Coding | Math/Sci | Business | Total Chars | Issues |
|---------|--------|---------|-------|--------|----------|----------|-------------|--------|
| FreedomIntelligence/alpaca-gpt4-japanese | ❌ | 0 | 0 | 0 | 0 | 0 | 0 | dataset_not_found |
| FreedomIntelligence/sharegpt-japanese | ❌ | 0 | 0 | 0 | 0 | 0 | 0 | dataset_not_found |
| FreedomIntelligence/MMLU_Japanese | ✅ | 0 | 0 | 2 | 0 | 0 | 448 | None |
| shi3z/anthropic_hh_rlhf_japanese | ✅ | 0 | 0 | 0 | 0 | 0 | 184 | json_parse_error_arlhfj.jsonl: Expecting value: line 1 column 1 (char 0) |
| fujiki/japanese_hh-rlhf-49k | ❌ | 0 | 0 | 0 | 0 | 0 | 0 | dataset_not_found |
| nomic-ai/gpt4all-j-prompt-generations | ✅ | 0 | 0 | 8 | 0 | 2 | 3418 | None |
| teknium/GPTeacher-General-Instruct | ✅ | 0 | 0 | 2 | 0 | 0 | 916 | json_parse_error_gpt4-instruct-dedupe-only-dataset.json: Expecting value: line 1 column 1 (char 0), json_parse_error_gpt4-instruct-similarity-0.6-dataset.json: Expecting value: line 1 column 1 (char 0), json_parse_error_gpt4-instruct-similarity-0.7-dataset.json: Expecting value: line 1 column 1 (char 0), json_parse_error_gpt4-instruct-similarity-0.8-dataset.json: Expecting value: line 1 column 1 (char 0), json_parse_error_gpt4-instruct-similarity-0.9-dataset.json: Expecting value: line 1 column 1 (char 0) |
| ehartford/wizard_vicuna_70k_unfiltered | ✅ | 0 | 0 | 0 | 0 | 0 | 696 | json_parse_error_wizard_vicuna_dataset_unfiltered.json: Expecting value: line 1 column 1 (char 0) |
| OpenAssistant/oasst2 | ✅ | 0 | 4 | 8 | 0 | 2 | 21212 | None |
| open-orca/OpenOrca | ✅ | 0 | 0 | 8 | 0 | 4 | 23938 | None |
| Elizezen/japanese-nsfw-syosetsu-dataset | ✅ | 1 | 4 | 2 | 0 | 0 | 1426 | json_parse_error_nsfw_0.json: Expecting value: line 1 column 1 (char 0) |
| eliasalbouzidi/NSFW-Safe-Dataset | ✅ | 0 | 2 | 8 | 0 | 2 | 5170 | None |
| RyokoExtra/JapaneseGoblin | ✅ | 1 | 0 | 4 | 0 | 0 | 5596 | json_parse_error_touhou.dump.json: Expecting value: line 1 column 1 (char 0) |

## Content Analysis Summary

### Primary Use Distribution
#### Reasoning Datasets (4 datasets)
- NSFW keywords: 0
- Coding keywords: 14
- Math/Science keywords: 0
- Business/MCP keywords: 4

#### Safety Datasets (1 datasets)
- NSFW keywords: 0
- Coding keywords: 0
- Math/Science keywords: 0
- Business/MCP keywords: 0

#### Coding Datasets (3 datasets)
- NSFW keywords: 4
- Coding keywords: 18
- Math/Science keywords: 0
- Business/MCP keywords: 4

#### Nsfw Datasets (2 datasets)
- NSFW keywords: 6
- Coding keywords: 10
- Math/Science keywords: 0
- Business/MCP keywords: 2
