# Dataset Quality Audit (2025-11-27 18:21:13)

| Dataset | Samples | Target OK | Logic % | NSFW % | Diversity | Avg Prompt | Avg Response | Issues |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | --- |
| deep_research_raw | 12,597 | [OK] | 0.00% | 0.00% | 0.0018 | 40.7 | 0.0 | logic_consistency_drop, nsfw_coverage_low |
| deep_research_cleansed | 0 | [NG] | 0.00% | 0.00% | 0.0000 | 0.0 | 0.0 | below_target_samples, empty_file, logic_consistency_drop, nsfw_coverage_low |
| pairwise_dataset | 12,597 | [NG] | 0.00% | 0.00% | 0.0018 | 40.7 | 213.4 | below_target_samples, logic_consistency_drop, nsfw_coverage_low |
| thinking_sft_dataset | 1,441 | [NG] | 0.00% | 0.07% | 0.3331 | 28800.7 | 57666.6 | below_target_samples, logic_consistency_drop, nsfw_coverage_low |

## Detailed Notes
### deep_research_raw
- Path: `D:\webdataset\aegis_v2.0\deep_research_thinking_dataset.jsonl`
- File size: 12,700,297 bytes
- Unique prompts ratio: 0.0018
- Issues: ['logic_consistency_drop', 'nsfw_coverage_low']

### deep_research_cleansed
- Path: `D:\webdataset\aegis_v2.0\deep_research_thinking_dataset_cleansed.jsonl`
- File size: 0 bytes
- Unique prompts ratio: 0.0000
- Issues: ['below_target_samples', 'empty_file', 'logic_consistency_drop', 'nsfw_coverage_low']

### pairwise_dataset
- Path: `D:\webdataset\aegis_v2.0\pairwise_dataset.jsonl`
- File size: 12,700,297 bytes
- Unique prompts ratio: 0.0018
- Four-class distribution: {'ALLOW': 12597}
- Quality score avg: 0.600 [0.60, 0.60]
- Issues: ['below_target_samples', 'logic_consistency_drop', 'nsfw_coverage_low']

### thinking_sft_dataset
- Path: `D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl`
- File size: 160,792,394 bytes
- Unique prompts ratio: 0.3331
- Four-class distribution: {'ALLOW': 1}
- Domain distribution (top 5): {'government': 1, 'unknown': 1440}
- Issues: ['below_target_samples', 'logic_consistency_drop', 'nsfw_coverage_low']
