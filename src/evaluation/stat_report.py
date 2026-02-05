"""ANOVA and Cohen's d reporting + visualization for model comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def load_scores(path: Path) -> pd.DataFrame:
    if path.suffix in {'.csv', '.tsv'}:
        sep = ',' if path.suffix == '.csv' else '\t'
        df = pd.read_csv(path, sep=sep)
    else:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            rows = []
            for model, scores in data.items():
                for score in scores:
                    rows.append({'model': model, 'score': score})
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(data)
    if 'model' not in df.columns or 'score' not in df.columns:
        raise ValueError('Input must contain model and score columns')
    return df


def one_way_anova(groups: Dict[str, np.ndarray]) -> Tuple[float, float]:
    all_scores = np.concatenate(list(groups.values()))
    grand_mean = np.mean(all_scores)
    ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    ss_within = sum(((v - np.mean(v)) ** 2).sum() for v in groups.values())
    df_between = len(groups) - 1
    df_within = len(all_scores) - len(groups)
    ms_between = ss_between / df_between if df_between else 0
    ms_within = ss_within / df_within if df_within else 0
    f_stat = ms_between / ms_within if ms_within else 0
    try:
        from scipy.stats import f  # type: ignore
        p_value = f.sf(f_stat, df_between, df_within)
    except Exception:
        p_value = float('nan')
    return f_stat, p_value


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2) if (n1 + n2 - 2) else 0
    return (np.mean(a) - np.mean(b)) / np.sqrt(pooled) if pooled else 0


def plot_distributions(df: pd.DataFrame, outdir: Path) -> None:
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model', y='score')
    plt.title('Model Score Distribution (Boxplot)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / 'score_boxplot.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='model', y='score', inner='quartile')
    plt.title('Model Score Distribution (Violin)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / 'score_violin.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x='model', y='score', estimator=np.mean, errorbar='sd')
    plt.title('Model Mean Score (±SD)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / 'score_mean_sd.png', dpi=200)
    plt.close()


def plot_html(df: pd.DataFrame, outdir: Path) -> None:
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        'Boxplot', 'Violin', 'Mean ± SD'
    ))

    box = px.box(df, x='model', y='score').data
    for trace in box:
        fig.add_trace(trace, row=1, col=1)

    violin = px.violin(df, x='model', y='score', box=True, points='outliers').data
    for trace in violin:
        fig.add_trace(trace, row=1, col=2)

    means = df.groupby('model')['score'].agg(['mean', 'std']).reset_index()
    fig.add_trace(
        go.Scatter(
            x=means['model'],
            y=means['mean'],
            error_y=dict(type='data', array=means['std']),
            mode='markers',
            name='Mean ± SD',
        ),
        row=1, col=3
    )

    fig.update_layout(height=450, width=1200, title_text='Model Score Summary')
    html_path = outdir / 'score_report.html'
    pio.write_html(fig, file=str(html_path), auto_open=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate ANOVA and Cohen\'s d report + plots')
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', default='reports/stats')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--html', action='store_true', help='Generate Plotly HTML report')
    args = parser.parse_args()

    df = load_scores(Path(args.input))
    groups = {m: g['score'].to_numpy() for m, g in df.groupby('model')}

    f_stat, p_value = one_way_anova(groups)
    models = list(groups.keys())
    d_matrix = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            d_matrix[f'{m1} vs {m2}'] = cohens_d(groups[m1], groups[m2])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_md = outdir / 'anova_cohensd_report.md'
    summary_json = outdir / 'anova_cohensd_summary.json'

    report_md.write_text(
        '# ANOVA / Cohen\'s d Report\n\n'
        f'- F-statistic: {f_stat:.4f}\n'
        f'- p-value: {p_value}\n\n'
        '## Pairwise Cohen\'s d\n' + '\n'.join([f'- {k}: {v:.4f}' for k, v in d_matrix.items()]) + '\n',
        encoding='utf-8',
    )
    summary_json.write_text(json.dumps({'f_stat': f_stat, 'p_value': p_value, 'cohens_d': d_matrix}, indent=2), encoding='utf-8')

    if not args.no_plots:
        plot_distributions(df, outdir)
    if args.html:
        plot_html(df, outdir)


if __name__ == '__main__':
    main()
