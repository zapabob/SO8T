#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合パイプラインStreamlitダッシュボード

統合パイプラインとすべてのWebスクレイピングの実行状況・進捗・ブラウジング風景を表示するダッシュボード
"""

import sys
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ユーティリティをインポート
from scripts.dashboard.unified_pipeline_dashboard_utils import (
    load_unified_pipeline_checkpoint,
    load_parallel_scraping_status,
    load_browser_screenshots,
    load_data_processing_status,
    load_ab_test_status,
    load_pipeline_logs,
    calculate_phase_progress,
    get_phase_status_color,
    format_duration
)

# サイバーパンク風CSS
CYBERPUNK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #0a0a0a 100%);
        color: #00ff41;
        font-family: 'Orbitron', monospace;
    }
    
    .main .block-container {
        background: rgba(0, 0, 0, 0.8);
        border: 2px solid #00ff41;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
    }
    
    h1, h2, h3 {
        color: #00ff41;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        font-family: 'Orbitron', monospace;
        font-weight: 900;
    }
    
    .stMetric {
        background: rgba(0, 255, 65, 0.1);
        border: 1px solid #00ff41;
        border-radius: 5px;
        padding: 1rem;
    }
    
    .stMetric label {
        color: #00ff41;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #00ff41;
        text-shadow: 0 0 5px rgba(0, 255, 65, 0.5);
    }
    
    .status-running {
        color: #00ff41;
        animation: pulse 2s infinite;
    }
    
    .status-completed {
        color: #00ff41;
    }
    
    .status-failed {
        color: #ff0040;
        animation: blink 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
</style>
"""

# ページ設定
st.set_page_config(
    page_title="SO8T統合パイプラインダッシュボード",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSSを適用
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


def load_config(config_path: Path) -> Dict:
    """設定ファイルを読み込み"""
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return {}


def create_progress_gauge(value: float, max_value: float, title: str, color: str = '#00ff41') -> go.Figure:
    """進捗ゲージチャートを作成"""
    percentage = (value / max_value * 100) if max_value > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': color, 'size': 16}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': color},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 0, 0, 0.3)'},
                {'range': [50, 80], 'color': 'rgba(0, 255, 65, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(0, 255, 65, 0.4)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': color, 'family': 'Orbitron'},
        height=250
    )
    
    return fig


def render_pipeline_status_tab(checkpoint_dir: Path, config: Dict):
    """統合パイプライン状態タブをレンダリング"""
    st.markdown('<h2>🚀 UNIFIED PIPELINE STATUS</h2>', unsafe_allow_html=True)
    
    # チェックポイントを読み込み
    checkpoint = load_unified_pipeline_checkpoint(checkpoint_dir)
    
    if not checkpoint:
        st.info("No checkpoint data available. Pipeline may not be running.")
        return
    
    session_id = checkpoint.get('session_id', 'unknown')
    current_phase = checkpoint.get('current_phase', 'unknown')
    phase_progress = checkpoint.get('phase_progress', {})
    timestamp = checkpoint.get('timestamp', '')
    
    # セッション情報
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session ID", session_id)
    with col2:
        st.metric("Current Phase", current_phase.replace('_', ' ').title())
    with col3:
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                st.metric("Last Update", dt.strftime('%Y-%m-%d %H:%M:%S'))
            except:
                st.metric("Last Update", timestamp)
    
    st.markdown("---")
    
    # Phase 1: 並列DeepResearch Webスクレイピング
    st.markdown('<h3>Phase 1: Parallel DeepResearch Web Scraping</h3>', unsafe_allow_html=True)
    phase1_data = phase_progress.get('phase1_parallel_scraping', {})
    phase1_status = phase1_data.get('status', 'pending')
    phase1_color = get_phase_status_color(phase1_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase1_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase1_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase1_color}; font-size: 20px;">Status: <strong>{phase1_status.upper()}</strong></p>', unsafe_allow_html=True)
        if phase1_data.get('process_id'):
            st.write(f"Process ID: {phase1_data['process_id']}")
        if phase1_data.get('started_at'):
            st.write(f"Started: {phase1_data['started_at']}")
        if phase1_data.get('note'):
            st.write(f"Note: {phase1_data['note']}")
    
    st.markdown("---")
    
    # Phase 2: SO8T全自動データ処理
    st.markdown('<h3>Phase 2: SO8T Auto Data Processing</h3>', unsafe_allow_html=True)
    phase2_data = phase_progress.get('phase2_data_processing', {})
    phase2_status = phase2_data.get('status', 'pending')
    phase2_color = get_phase_status_color(phase2_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase2_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase2_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase2_color}; font-size: 20px;">Status: <strong>{phase2_status.upper()}</strong></p>', unsafe_allow_html=True)
        if phase2_data.get('completed_at'):
            st.write(f"Completed: {phase2_data['completed_at']}")
    
    st.markdown("---")
    
    # Phase 3: SO8T完全統合A/Bテスト
    st.markdown('<h3>Phase 3: SO8T Complete A/B Test</h3>', unsafe_allow_html=True)
    phase3_data = phase_progress.get('phase3_ab_test', {})
    phase3_status = phase3_data.get('status', 'pending')
    phase3_color = get_phase_status_color(phase3_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase3_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase3_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase3_color}; font-size: 20px;">Status: <strong>{phase3_status.upper()}</strong></p>', unsafe_allow_html=True)
        if phase3_data.get('completed_at'):
            st.write(f"Completed: {phase3_data['completed_at']}")


def render_scraping_status_tab(output_dir: Path, screenshots_dir: Path):
    """Webスクレイピング状態タブをレンダリング"""
    st.markdown('<h2>🔍 WEB SCRAPING STATUS</h2>', unsafe_allow_html=True)
    
    # スクレイピング状態を読み込み
    scraping_status = load_parallel_scraping_status(output_dir)
    browser_status = scraping_status.get('browser_status', {})
    total_samples = scraping_status.get('total_samples', 0)
    
    # 全体統計
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{total_samples:,}")
    with col2:
        active_browsers = sum(1 for s in browser_status.values() if s.get('status') == 'active')
        st.metric("Active Browsers", active_browsers)
    with col3:
        completed_browsers = sum(1 for s in browser_status.values() if s.get('status') == 'completed')
        st.metric("Completed Browsers", completed_browsers)
    with col4:
        total_browsers = len(browser_status)
        st.metric("Total Browsers", total_browsers)
    
    st.markdown("---")
    
    # ブラウザ状態テーブル
    if browser_status:
        st.markdown('<h3>Browser Status</h3>', unsafe_allow_html=True)
        browser_df_data = []
        for browser_num, status in browser_status.items():
            browser_df_data.append({
                'Browser': browser_num,
                'Status': status.get('status', 'unknown'),
                'Current Keyword': status.get('current_keyword', 'None'),
                'Samples Collected': status.get('samples_collected', 0),
                'Last Activity': status.get('last_activity', 'None')
            })
        
        browser_df = pd.DataFrame(browser_df_data)
        st.dataframe(browser_df, use_container_width=True)
        
        # 状態別ブラウザ数の可視化
        col1, col2 = st.columns(2)
        with col1:
            status_counts = {}
            for status in browser_status.values():
                s = status.get('status', 'unknown')
                status_counts[s] = status_counts.get(s, 0) + 1
            
            if status_counts:
                st.bar_chart(status_counts)
        
        with col2:
            samples_by_browser = {
                f"Browser {num}": status.get('samples_collected', 0)
                for num, status in browser_status.items()
            }
            if samples_by_browser:
                st.bar_chart(samples_by_browser)
    
    st.markdown("---")
    
    # ブラウザスクリーンショット表示
    st.markdown('<h3>🌐 Browser Screenshots</h3>', unsafe_allow_html=True)
    
    screenshots = load_browser_screenshots(screenshots_dir, max_count=20)
    
    if screenshots:
        # ブラウザごとにグループ化
        browser_screenshots = {}
        for screenshot in screenshots:
            browser_num = screenshot.get('browser_num')
            if browser_num is not None:
                if browser_num not in browser_screenshots:
                    browser_screenshots[browser_num] = []
                browser_screenshots[browser_num].append(screenshot)
        
        # 各ブラウザの最新スクリーンショットを表示
        if browser_screenshots:
            cols_per_row = 2
            browsers = sorted(browser_screenshots.keys())
            
            for row_start in range(0, len(browsers), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    browser_idx = row_start + col_idx
                    if browser_idx < len(browsers):
                        browser_num = browsers[browser_idx]
                        latest_screenshot = browser_screenshots[browser_num][0]
                        
                        with cols[col_idx]:
                            st.markdown(f"**Browser {browser_num}**")
                            st.image(
                                latest_screenshot['image'],
                                use_container_width=True,
                                caption=f"Browser {browser_num} - {latest_screenshot['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                            )
        else:
            # ブラウザ番号がない場合は時系列で表示
            latest_screenshot = screenshots[0]
            st.image(
                latest_screenshot['image'],
                use_container_width=True,
                caption=f"Latest Screenshot - {latest_screenshot['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # 過去のスクリーンショットをサムネイル表示
            if len(screenshots) > 1:
                st.markdown('<h4>Screenshot History</h4>', unsafe_allow_html=True)
                cols = st.columns(min(len(screenshots) - 1, 4))
                for i, screenshot in enumerate(screenshots[1:5]):
                    with cols[i % 4]:
                        st.image(
                            screenshot['image'],
                            use_container_width=True,
                            caption=screenshot['timestamp'].strftime('%H:%M:%S')
                        )
    else:
        st.info("No browser screenshots available. Browser capture may not be active.")


def render_data_processing_tab(checkpoint_dir: Path):
    """データ処理状態タブをレンダリング"""
    st.markdown('<h2>📊 DATA PROCESSING STATUS</h2>', unsafe_allow_html=True)
    
    # データ処理状態を読み込み
    processing_status = load_data_processing_status(checkpoint_dir)
    
    # Phase 1: データクレンジング
    st.markdown('<h3>Phase 1: Data Cleaning</h3>', unsafe_allow_html=True)
    phase1_data = processing_status.get('phase1_data_cleaning', {})
    phase1_status = phase1_data.get('status', 'pending')
    phase1_color = get_phase_status_color(phase1_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase1_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase1_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase1_color}; font-size: 20px;">Status: <strong>{phase1_status.upper()}</strong></p>', unsafe_allow_html=True)
        st.metric("Processed Samples", f"{phase1_data.get('samples', 0):,}")
    
    st.markdown("---")
    
    # Phase 2: 漸次ラベル付け
    st.markdown('<h3>Phase 2: Incremental Labeling</h3>', unsafe_allow_html=True)
    phase2_data = processing_status.get('phase2_incremental_labeling', {})
    phase2_status = phase2_data.get('status', 'pending')
    phase2_color = get_phase_status_color(phase2_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase2_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase2_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase2_color}; font-size: 20px;">Status: <strong>{phase2_status.upper()}</strong></p>', unsafe_allow_html=True)
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Labeled Samples", f"{phase2_data.get('samples', 0):,}")
        with col2_2:
            quality = phase2_data.get('quality', 0.0)
            st.metric("Average Quality", f"{quality:.3f}")
    
    st.markdown("---")
    
    # Phase 3: 四値分類
    st.markdown('<h3>Phase 3: Quadruple Classification</h3>', unsafe_allow_html=True)
    phase3_data = processing_status.get('phase3_quadruple_classification', {})
    phase3_status = phase3_data.get('status', 'pending')
    phase3_color = get_phase_status_color(phase3_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase3_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase3_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase3_color}; font-size: 20px;">Status: <strong>{phase3_status.upper()}</strong></p>', unsafe_allow_html=True)
        st.metric("Classified Samples", f"{phase3_data.get('samples', 0):,}")


def render_ab_test_tab(checkpoint_dir: Path, results_dir: Path):
    """A/Bテスト状態タブをレンダリング"""
    st.markdown('<h2>🧪 A/B TEST STATUS</h2>', unsafe_allow_html=True)
    
    # A/Bテスト状態を読み込み
    ab_test_status = load_ab_test_status(checkpoint_dir, results_dir)
    
    # Phase 1: Model AのGGUF変換
    st.markdown('<h3>Phase 1: Model A GGUF Conversion</h3>', unsafe_allow_html=True)
    phase1_data = ab_test_status.get('phase1_model_a_gguf', {})
    phase1_status = phase1_data.get('status', 'pending')
    phase1_color = get_phase_status_color(phase1_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase1_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase1_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase1_color}; font-size: 20px;">Status: <strong>{phase1_status.upper()}</strong></p>', unsafe_allow_html=True)
        if phase1_data.get('gguf_files'):
            st.write("GGUF Files:")
            for quant_type, file_path in phase1_data['gguf_files'].items():
                st.write(f"  - {quant_type}: {Path(file_path).name}")
    
    st.markdown("---")
    
    # Phase 2: SO8T再学習
    st.markdown('<h3>Phase 2: SO8T Retraining</h3>', unsafe_allow_html=True)
    phase2_data = ab_test_status.get('phase2_train_model_b', {})
    phase2_status = phase2_data.get('status', 'pending')
    phase2_color = get_phase_status_color(phase2_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase2_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase2_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase2_color}; font-size: 20px;">Status: <strong>{phase2_status.upper()}</strong></p>', unsafe_allow_html=True)
        if phase2_data.get('trained_model_path'):
            st.write(f"Trained Model: {Path(phase2_data['trained_model_path']).name}")
    
    st.markdown("---")
    
    # Phase 3: Model BのGGUF変換
    st.markdown('<h3>Phase 3: Model B GGUF Conversion</h3>', unsafe_allow_html=True)
    phase3_data = ab_test_status.get('phase3_model_b_gguf', {})
    phase3_status = phase3_data.get('status', 'pending')
    phase3_color = get_phase_status_color(phase3_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase3_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase3_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase3_color}; font-size: 20px;">Status: <strong>{phase3_status.upper()}</strong></p>', unsafe_allow_html=True)
        if phase3_data.get('gguf_files'):
            st.write("GGUF Files:")
            for quant_type, file_path in phase3_data['gguf_files'].items():
                st.write(f"  - {quant_type}: {Path(file_path).name}")
    
    st.markdown("---")
    
    # Phase 4: Ollamaインポート
    st.markdown('<h3>Phase 4: Ollama Import</h3>', unsafe_allow_html=True)
    phase4_data = ab_test_status.get('phase4_ollama_import', {})
    phase4_status = phase4_data.get('status', 'pending')
    phase4_color = get_phase_status_color(phase4_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase4_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase4_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase4_color}; font-size: 20px;">Status: <strong>{phase4_status.upper()}</strong></p>', unsafe_allow_html=True)
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            model_a_imported = phase4_data.get('model_a_imported', False)
            st.metric("Model A Imported", "Yes" if model_a_imported else "No")
        with col2_2:
            model_b_imported = phase4_data.get('model_b_imported', False)
            st.metric("Model B Imported", "Yes" if model_b_imported else "No")
    
    st.markdown("---")
    
    # Phase 5: A/Bテスト実行
    st.markdown('<h3>Phase 5: A/B Test Execution</h3>', unsafe_allow_html=True)
    phase5_data = ab_test_status.get('phase5_ab_test', {})
    phase5_status = phase5_data.get('status', 'pending')
    phase5_color = get_phase_status_color(phase5_status)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        progress = calculate_phase_progress(phase5_data)
        fig = create_progress_gauge(progress * 100, 100, "Progress", phase5_color)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<p style="color: {phase5_color}; font-size: 20px;">Status: <strong>{phase5_status.upper()}</strong></p>', unsafe_allow_html=True)
    
    # A/Bテスト結果表示
    results = ab_test_status.get('results')
    if results:
        st.markdown("---")
        st.markdown('<h3>A/B Test Results</h3>', unsafe_allow_html=True)
        
        model_a_metrics = results.get('model_a', {})
        model_b_metrics = results.get('model_b', {})
        comparison = results.get('comparison', {})
        
        # メトリクス比較
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model A Accuracy", f"{model_a_metrics.get('accuracy', 0):.4f}")
            st.metric("Model A F1 Macro", f"{model_a_metrics.get('f1_macro', 0):.4f}")
            st.metric("Model A Latency", f"{model_a_metrics.get('avg_latency', 0):.4f}s")
        
        with col2:
            st.metric("Model B Accuracy", f"{model_b_metrics.get('accuracy', 0):.4f}")
            st.metric("Model B F1 Macro", f"{model_b_metrics.get('f1_macro', 0):.4f}")
            st.metric("Model B Latency", f"{model_b_metrics.get('avg_latency', 0):.4f}s")
        
        with col3:
            accuracy_improvement = comparison.get('accuracy_improvement', 0)
            f1_improvement = comparison.get('f1_macro_improvement', 0)
            latency_change = comparison.get('latency_change', 0)
            
            st.metric("Accuracy Improvement", f"{accuracy_improvement:+.4f}", delta=f"{accuracy_improvement:+.4f}")
            st.metric("F1 Macro Improvement", f"{f1_improvement:+.4f}", delta=f"{f1_improvement:+.4f}")
            st.metric("Latency Change", f"{latency_change:+.4f}s", delta=f"{latency_change:+.4f}s")
        
        # メトリクス比較チャート
        metrics_data = {
            'Metric': ['Accuracy', 'F1 Macro', 'Latency (s)'],
            'Model A': [
                model_a_metrics.get('accuracy', 0),
                model_a_metrics.get('f1_macro', 0),
                model_a_metrics.get('avg_latency', 0)
            ],
            'Model B': [
                model_b_metrics.get('accuracy', 0),
                model_b_metrics.get('f1_macro', 0),
                model_b_metrics.get('avg_latency', 0)
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.set_index('Metric')
        st.bar_chart(metrics_df)


def render_logs_tab(log_dir: Path):
    """ログ表示タブをレンダリング"""
    st.markdown('<h2>📝 PIPELINE LOGS</h2>', unsafe_allow_html=True)
    
    # ログを読み込み
    log_lines = load_pipeline_logs(log_dir, max_lines=200)
    
    if log_lines:
        # ログを表示（最新のものから）
        log_text = ''.join(log_lines)
        st.code(log_text, language='text')
        
        # ログファイルの更新時刻
        log_file = log_dir / "unified_master_pipeline.log"
        if log_file.exists():
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            st.caption(f"Last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No log file available.")


def main():
    """メイン関数"""
    # タイトル
    st.markdown('<h1>🚀 SO8T UNIFIED PIPELINE DASHBOARD</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00ffff; font-size: 18px;">Real-time monitoring of unified pipeline and web scraping progress</p>', unsafe_allow_html=True)
    
    # 設定ファイルを読み込み
    config_path = PROJECT_ROOT / "configs" / "unified_pipeline_dashboard_config.yaml"
    config = load_config(config_path)
    
    # デフォルト設定
    checkpoint_dir = Path(config.get('checkpoint_dir', 'D:/webdataset/checkpoints/unified_master_pipeline'))
    output_dir = Path(config.get('output_dir', 'D:/webdataset/processed'))
    screenshots_dir = Path(config.get('screenshots_dir', output_dir / 'screenshots'))
    results_dir = Path(config.get('results_dir', 'D:/webdataset/ab_test_results'))
    log_dir = Path(config.get('log_dir', PROJECT_ROOT / 'logs'))
    
    # 自動更新設定
    col1, col2, col3 = st.columns(3)
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    with col2:
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 5)
    with col3:
        if st.button("🔄 Manual Refresh"):
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    # タブを作成
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Unified Pipeline",
        "🔍 Web Scraping",
        "📊 Data Processing",
        "🧪 A/B Test",
        "📝 Logs"
    ])
    
    with tab1:
        render_pipeline_status_tab(checkpoint_dir, config)
    
    with tab2:
        render_scraping_status_tab(output_dir, screenshots_dir)
    
    with tab3:
        render_data_processing_tab(checkpoint_dir)
    
    with tab4:
        render_ab_test_tab(checkpoint_dir, results_dir)
    
    with tab5:
        render_logs_tab(log_dir)
    
    # 自動更新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()














