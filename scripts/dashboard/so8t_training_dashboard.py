#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T再学習進捗Streamlitダッシュボード

リアルタイムで学習進捗、システムメトリクス、学習曲線を可視化
"""

import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.dashboard_utils import (
    load_progress_logs,
    load_session_info,
    load_checkpoint_info,
    calculate_progress,
    estimate_remaining_time,
    get_latest_session,
    get_elapsed_time
)

# ページ設定
st.set_page_config(
    page_title="SO8T再学習進捗ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設定読み込み
@st.cache_data
def load_config():
    """設定ファイルを読み込み"""
    config_path = PROJECT_ROOT / "configs" / "dashboard_config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {
        'checkpoint_base_dir': 'D:/webdataset/checkpoints/training',
        'refresh_interval': 5,
        'port': 8501,
        'gpu_temp_warning': 75
    }


def create_gauge_chart(value: float, title: str, max_value: float = 100.0, 
                       warning_threshold: Optional[float] = None) -> go.Figure:
    """ゲージチャートを作成"""
    # 色の決定
    if warning_threshold and value >= warning_threshold:
        color = 'red'
    elif value >= max_value * 0.8:
        color = 'orange'
    else:
        color = 'green'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_value * 0.6], 'color': "lightgray"},
                {'range': [max_value * 0.6, max_value * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': warning_threshold if warning_threshold else max_value
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    """メイン関数"""
    # タイトル
    st.title("📊 SO8T再学習進捗ダッシュボード")
    
    # 設定読み込み
    config = load_config()
    checkpoint_base = Path(config.get('checkpoint_base_dir', 'D:/webdataset/checkpoints/training'))
    refresh_interval = config.get('refresh_interval', 5)
    gpu_temp_warning = config.get('gpu_temp_warning', 75)
    
    # サイドバー
    with st.sidebar:
        st.header("設定")
        
        # 更新間隔設定
        refresh_interval = st.slider("更新間隔（秒）", 1, 30, refresh_interval)
        
        # 手動更新ボタン
        if st.button("🔄 手動更新"):
            st.rerun()
        
        # セッション選択
        st.header("セッション選択")
        session_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()] if checkpoint_base.exists() else []
        session_names = [d.name for d in session_dirs]
        
        if session_names:
            selected_session = st.selectbox("セッション", session_names, index=0)
            session_dir = checkpoint_base / selected_session
        else:
            st.warning("セッションが見つかりません")
            session_dir = None
    
    if session_dir is None or not session_dir.exists():
        st.error(f"セッションディレクトリが見つかりません: {session_dir}")
        return
    
    # データ読み込み
    session_info = load_session_info(session_dir)
    progress_logs_dir = session_dir / "progress_logs"
    progress_logs = load_progress_logs(progress_logs_dir)
    checkpoint_info = load_checkpoint_info(session_dir)
    
    # セッション情報セクション
    st.header("📋 セッション情報")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("セッションID", session_info.get('session_id', 'N/A') if session_info else 'N/A')
    with col2:
        status = session_info.get('status', 'unknown') if session_info else 'unknown'
        status_color = {
            'running': '🟢',
            'completed': '✅',
            'interrupted': '⚠️',
            'unknown': '❓'
        }.get(status, '❓')
        st.metric("ステータス", f"{status_color} {status}")
    with col3:
        start_time = session_info.get('start_time', 'N/A') if session_info else 'N/A'
        if start_time != 'N/A':
            try:
                dt = datetime.fromisoformat(start_time)
                start_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        st.metric("開始時刻", start_time)
    with col4:
        elapsed_time = get_elapsed_time(session_info)
        st.metric("経過時間", elapsed_time if elapsed_time else 'N/A')
    
    # 学習進捗セクション
    st.header("📈 学習進捗")
    
    if session_info and progress_logs:
        current_epoch = session_info.get('current_epoch', 0)
        current_step = session_info.get('current_step', 0)
        total_steps = session_info.get('total_steps', 0)
        best_loss = session_info.get('best_loss', float('inf'))
        
        # 最新のログから情報を取得
        latest_log = progress_logs[-1] if progress_logs else {}
        current_loss = latest_log.get('loss', 0.0)
        learning_rate = latest_log.get('learning_rate', 0.0)
        
        # 進捗率計算
        progress = calculate_progress(current_step, total_steps)
        
        # 残り時間推定
        remaining_time = estimate_remaining_time(progress_logs, current_step, total_steps)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            num_epochs = session_info.get('num_epochs', 3)
            st.metric("エポック", f"{current_epoch}/{num_epochs}")
        with col2:
            st.metric("ステップ", f"{current_step:,}/{total_steps:,}", f"{progress*100:.1f}%")
        with col3:
            st.metric("損失値", f"{current_loss:.4f}", f"Best: {best_loss:.4f}" if best_loss != float('inf') else None)
        with col4:
            st.metric("学習率", f"{learning_rate:.2e}")
        with col5:
            st.metric("推定残り時間", remaining_time if remaining_time else 'N/A')
        
        # 進捗バー
        st.progress(progress)
    else:
        st.info("学習進捗データがありません")
    
    # システムメトリクスセクション
    st.header("💻 システムメトリクス")
    
    if progress_logs:
        latest_log = progress_logs[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            cpu_usage = latest_log.get('cpu_usage', 0.0)
            fig_cpu = create_gauge_chart(cpu_usage, "CPU使用率", max_value=100.0)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            memory_usage = latest_log.get('memory_usage', 0.0)
            fig_memory = create_gauge_chart(memory_usage, "メモリ使用率", max_value=100.0)
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col3:
            gpu_usage = latest_log.get('gpu_usage', 0.0)
            fig_gpu = create_gauge_chart(gpu_usage, "GPU使用率", max_value=100.0)
            st.plotly_chart(fig_gpu, use_container_width=True)
        
        with col4:
            gpu_memory_usage = latest_log.get('gpu_memory_usage', 0.0)
            fig_gpu_mem = create_gauge_chart(gpu_memory_usage, "GPUメモリ使用率", max_value=100.0)
            st.plotly_chart(fig_gpu_mem, use_container_width=True)
        
        with col5:
            gpu_temp = latest_log.get('gpu_temperature', 0.0)
            fig_gpu_temp = create_gauge_chart(gpu_temp, "GPU温度", max_value=100.0, warning_threshold=gpu_temp_warning)
            st.plotly_chart(fig_gpu_temp, use_container_width=True)
    else:
        st.info("システムメトリクスデータがありません")
    
    # 学習曲線セクション
    st.header("📊 学習曲線")
    
    if progress_logs:
        # データフレーム作成
        df = pd.DataFrame(progress_logs)
        
        # タイムスタンプをdatetimeに変換
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 損失値の推移
            if 'loss' in df.columns and 'step' in df.columns:
                fig_loss = px.line(df, x='step', y='loss', 
                                   title='損失値の推移',
                                   labels={'step': 'ステップ', 'loss': '損失値'})
                fig_loss.update_layout(height=400)
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.info("損失値データがありません")
        
        with col2:
            # 学習率の推移
            if 'learning_rate' in df.columns and 'step' in df.columns:
                fig_lr = px.line(df, x='step', y='learning_rate',
                                title='学習率の推移',
                                labels={'step': 'ステップ', 'learning_rate': '学習率'})
                fig_lr.update_layout(height=400)
                st.plotly_chart(fig_lr, use_container_width=True)
            else:
                st.info("学習率データがありません")
        
        # システムメトリクスの推移
        st.subheader("システムメトリクスの推移")
        metric_cols = ['cpu_usage', 'memory_usage', 'gpu_usage', 'gpu_memory_usage', 'gpu_temperature']
        available_metrics = [col for col in metric_cols if col in df.columns]
        
        if available_metrics and 'timestamp' in df.columns:
            fig_metrics = go.Figure()
            for metric in available_metrics:
                fig_metrics.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title()
                ))
            fig_metrics.update_layout(
                title='システムメトリクスの推移',
                xaxis_title='時刻',
                yaxis_title='使用率/温度',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            st.info("システムメトリクスデータがありません")
    else:
        st.info("学習曲線データがありません")
    
    # チェックポイント情報セクション
    st.header("💾 チェックポイント情報")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("最新チェックポイント")
        if checkpoint_info['rolling_checkpoints']:
            latest_checkpoint = Path(checkpoint_info['rolling_checkpoints'][0])
            st.write(f"**ファイル名**: {latest_checkpoint.name}")
            st.write(f"**パス**: {latest_checkpoint}")
            try:
                mtime = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
                st.write(f"**更新時刻**: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                pass
        elif checkpoint_info['final_checkpoint']:
            final_checkpoint = Path(checkpoint_info['final_checkpoint'])
            st.write(f"**ファイル名**: {final_checkpoint.name}")
            st.write(f"**パス**: {final_checkpoint}")
        else:
            st.info("チェックポイントが見つかりません")
    
    with col2:
        st.subheader("チェックポイント統計")
        st.metric("ローリングストック", f"{len(checkpoint_info['rolling_checkpoints'])}/5")
        st.metric("最終チェックポイント", "あり" if checkpoint_info['final_checkpoint'] else "なし")
        st.metric("緊急チェックポイント", len(checkpoint_info['emergency_checkpoints']))
        st.metric("総チェックポイント数", checkpoint_info['total_count'])
    
    # 自動更新設定
    if st.checkbox("🔄 自動更新を有効化", value=True):
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == '__main__':
    main()

