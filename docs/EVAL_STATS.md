# Stats Report (ANOVA / Cohen\'s d)

Generate a statistical report and plots from CSV or JSON.

## Input formats
- CSV/TSV with columns: `model`, `score`
- JSON dict: `{ "ModelA": [0.8, 0.9], "ModelB": [0.75, 0.7] }`

## Usage
```bash
py -m src.eval.stat_report --input results/abc_testing/combined_scores.csv --outdir reports/stats
```

## HTML (Plotly)
```bash
py -m src.eval.stat_report --input results/abc_testing/combined_scores.csv --outdir reports/stats --html
```

## Outputs
- `anova_cohensd_report.md`
- `anova_cohensd_summary.json`
- `score_boxplot.png`
- `score_violin.png`
- `score_mean_sd.png`
- `score_report.html` (when --html)
