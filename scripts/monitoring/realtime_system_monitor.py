#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T リアルタイムシステム監視ダッシュボード
サイバーパンク風UIでGPU/CPU/メモリ/ログをリアルタイム監視
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import time
from collections import deque

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# dashboard_utilsからget_system_metricsをインポート
try:
    from scripts.dashboard.dashboard_utils import get_system_metrics
except ImportError:
    logger.error("Failed to import get_system_metrics from dashboard_utils")
    # フォールバック実装
    def get_system_metrics():
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'gpu_memory_usage': 0.0,
            'gpu_temperature': 0.0,
            'gpu_available': False
        }

# サイバーパンク風CSS（拡張版）
CYBERPUNK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #0a0a0a 100%);
        background-attachment: fixed;
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
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.5), 0 0 20px rgba(0, 255, 65, 0.3);
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        letter-spacing: 2px;
    }
    
    .stMetric {
        background: rgba(0, 255, 65, 0.1);
        border: 1px solid #00ff41;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
    }
    
    .stMetric label {
        color: #00ff41;
        font-family: 'Orbitron', monospace;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #00ff41;
        text-shadow: 0 0 5px rgba(0, 255, 65, 0.5);
        font-family: 'Orbitron', monospace;
        font-weight: 700;
    }
    
    .cyber-border {
        border: 2px solid #00ff41;
        border-radius: 10px;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.5);
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.3);
    }
    
    .glitch-text {
        position: relative;
        color: #00ff41;
        text-shadow: 
            2px 2px 0 #ff0040,
            -2px -2px 0 #00ffff;
        animation: glitch 0.3s infinite;
    }
    
    @keyframes glitch {
        0% { transform: translate(0); }
        20% { transform: translate(-2px, 2px); }
        40% { transform: translate(-2px, -2px); }
        60% { transform: translate(2px, 2px); }
        80% { transform: translate(2px, -2px); }
        100% { transform: translate(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .log-container {
        background: rgba(0, 0, 0, 0.9);
        border: 1px solid #00ff41;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        max-height: 400px;
        overflow-y: auto;
        color: #00ff41;
    }
    
    .log-line {
        padding: 2px 0;
        border-bottom: 1px solid rgba(0, 255, 65, 0.1);
    }
    
    .log-error {
        color: #ff0040;
    }
    
    .log-warning {
        color: #ffaa00;
    }
    
    .log-info {
        color: #00ffff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.5);
        border-bottom: 2px solid #00ff41;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00ff00;
        text-shadow: 0 0 5px #00ff00;
    }
</style>
"""

# ページ設定
st.set_page_config(
    page_title="SO8T Real-Time System Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS適用
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# セッション状態の初期化
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = deque(maxlen=100)  # 最新100件を保持
if 'update_count' not in st.session_state:
    st.session_state.update_count = 0
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()


def create_gauge_chart(
    value: float,
    title: str,
    max_value: float = 100.0,
    warning_threshold: Optional[float] = None,
    unit: str = "%"
) -> go.Figure:
    """サイバーパンク風ゲージチャートを作成"""
    # 値の検証とクランプ
    value = max(0.0, min(float(value), float(max_value)))
    
    # 色決定
    if warning_threshold is not None and value >= warning_threshold:
        color = '#ff0080'  # マゼンタ（警告）
        bg_color = 'rgba(255, 0, 128, 0.2)'
    elif value >= max_value * 0.8:
        color = '#ffff00'  # イエロー（注意）
        bg_color = 'rgba(255, 255, 0, 0.2)'
    else:
        color = '#00ff00'  # グリーン（正常）
        bg_color = 'rgba(0, 255, 0, 0.2)'
    
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': f"<span style='font-family: Orbitron; color: #00ffff; font-size: 16px;'>{title}</span>",
                'font': {'size': 16}
            },
            number={
                'font': {'family': 'Orbitron', 'size': 24, 'color': color},
                'valueformat': '.1f',
                'suffix': unit
            },
            gauge={
                'axis': {
                    'range': [None, max_value],
                    'tickcolor': '#00ffff',
                    'tickfont': {'family': 'Orbitron', 'color': '#00ffff', 'size': 10}
                },
                'bar': {
                    'color': color,
                    'line': {'color': '#ffffff', 'width': 2}
                },
                'bgcolor': 'rgba(0, 0, 0, 0.8)',
                'borderwidth': 2,
                'bordercolor': '#00ffff',
                'steps': [
                    {'range': [0, max_value * 0.6], 'color': bg_color},
                    {'range': [max_value * 0.6, max_value * 0.8], 'color': 'rgba(255, 255, 0, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#ff0080", 'width': 3},
                    'thickness': 0.8,
                    'value': warning_threshold if warning_threshold else max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'family': 'Orbitron', 'color': '#00ffff'}
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating gauge chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            height=220,
            title={'text': f'Error: {str(e)}'},
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        return fig


def create_timeseries_chart(metrics_history: deque, metric_key: str, title: str, color: str = '#00ff41') -> go.Figure:
    """時系列チャートを作成"""
    if len(metrics_history) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font={'family': 'Orbitron', 'color': '#00ffff'},
            height=300
        )
        return fig
    
    timestamps = [m.get('timestamp', '') for m in metrics_history]
    values = [m.get(metric_key, 0.0) for m in metrics_history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name=title,
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
        fill='tonexty',
        fillcolor=f'{color}33'
    ))
    
    fig.update_layout(
        title={
            'text': f"<span style='font-family: Orbitron; color: #00ffff;'>{title}</span>",
            'font': {'size': 16}
        },
        xaxis={
            'title': 'Time',
            'tickfont': {'family': 'Orbitron', 'color': '#00ffff', 'size': 10},
            'gridcolor': 'rgba(0, 255, 65, 0.2)'
        },
        yaxis={
            'title': 'Value',
            'tickfont': {'family': 'Orbitron', 'color': '#00ffff', 'size': 10},
            'gridcolor': 'rgba(0, 255, 65, 0.2)'
        },
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0.5)',
        font={'family': 'Orbitron', 'color': '#00ffff'},
        height=300,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig


def load_log_file(log_path: Path, max_lines: int = 100) -> List[str]:
    """ログファイルを読み込み（最新N行）"""
    if not log_path.exists():
        return []
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return lines[-max_lines:]  # 最新N行
    except Exception as e:
        logger.error(f"Failed to load log file {log_path}: {e}")
        return []


def format_log_line(line: str) -> str:
    """ログ行をフォーマット（色分け）"""
    line = line.strip()
    if not line:
        return ""
    
    # エラーレベルの検出
    if 'ERROR' in line.upper() or 'EXCEPTION' in line.upper():
        return f'<span class="log-error">{line}</span>'
    elif 'WARNING' in line.upper() or 'WARN' in line.upper():
        return f'<span class="log-warning">{line}</span>'
    elif 'INFO' in line.upper():
        return f'<span class="log-info">{line}</span>'
    else:
        return f'<span>{line}</span>'


def main():
    """メイン関数"""
    # タイトル
    st.markdown('<h1 class="glitch-text">⚡ SO8T REAL-TIME SYSTEM MONITOR ⚡</h1>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown('<h2>⚙️ CONTROL PANEL</h2>', unsafe_allow_html=True)
        
        # 更新間隔設定
        refresh_interval = st.slider("更新間隔（秒）", 1, 10, 2, help="ダッシュボードの自動更新間隔")
        
        # 自動更新チェックボックス
        auto_refresh = st.checkbox("🔄 AUTO REFRESH", value=True)
        
        # 手動更新ボタン
        if st.button("🔄 FORCE REFRESH", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # ログファイル設定
        st.markdown('<h3>📝 LOG FILES</h3>', unsafe_allow_html=True)
        
        # デフォルトログファイルパス
        default_logs = [
            "D:/webdataset/pipeline_logs/master_automated_pipeline.log",
            "logs/agent_runtime.log",
            "logs/error.log"
        ]
        
        log_paths_input = st.text_area(
            "ログファイルパス（1行1ファイル）",
            value="\n".join(default_logs),
            height=100,
            help="監視するログファイルのパスを1行1ファイルで入力"
        )
        
        log_paths = [Path(p.strip()) for p in log_paths_input.split('\n') if p.strip()]
        
        st.markdown("---")
        
        # セッション情報
        st.markdown('<h3>📊 SESSION INFO</h3>', unsafe_allow_html=True)
        st.markdown(f"**更新回数**: {st.session_state.update_count}")
        st.markdown(f"**最終更新**: {datetime.now().strftime('%H:%M:%S')}")
        
        # メトリクス履歴の保存先
        metrics_dir = Path("logs/realtime_monitor")
        metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # システムメトリクス取得
    try:
        metrics = get_system_metrics()
        metrics['timestamp'] = datetime.now().strftime('%H:%M:%S')
        
        # メトリクス履歴に追加
        st.session_state.metrics_history.append(metrics)
        
        # メトリクス履歴をファイルに保存（オプション）
        if len(st.session_state.metrics_history) % 10 == 0:  # 10回ごとに保存
            try:
                history_file = metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
                history_data = list(st.session_state.metrics_history)
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save metrics history: {e}")
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'gpu_memory_usage': 0.0,
            'gpu_temperature': 0.0,
            'gpu_available': False,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    
    # メインコンテンツ
    # タブでセクションを分ける
    tab1, tab2, tab3 = st.tabs([
        "📊 System Metrics",
        "📈 Charts",
        "📝 Logs"
    ])
    
    with tab1:
        # システムメトリクス表示
        st.markdown('<h2>💻 SYSTEM METRICS</h2>', unsafe_allow_html=True)
        
        # GPU情報
        if metrics.get('gpu_available', False):
            st.markdown('<h3>🎮 GPU</h3>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gpu_temp = metrics.get('gpu_temperature', 0.0)
                fig_temp = create_gauge_chart(
                    gpu_temp,
                    "GPU Temperature",
                    max_value=100.0,
                    warning_threshold=75.0,
                    unit="°C"
                )
                st.plotly_chart(fig_temp, use_container_width=True)
                st.metric("GPU Temperature", f"{gpu_temp:.1f}°C")
            
            with col2:
                gpu_usage = metrics.get('gpu_usage', 0.0)
                fig_gpu = create_gauge_chart(
                    gpu_usage,
                    "GPU Usage",
                    max_value=100.0,
                    warning_threshold=90.0
                )
                st.plotly_chart(fig_gpu, use_container_width=True)
                st.metric("GPU Usage", f"{gpu_usage:.1f}%")
            
            with col3:
                gpu_mem = metrics.get('gpu_memory_usage', 0.0)
                fig_gpu_mem = create_gauge_chart(
                    gpu_mem,
                    "GPU Memory",
                    max_value=100.0,
                    warning_threshold=90.0
                )
                st.plotly_chart(fig_gpu_mem, use_container_width=True)
                st.metric("GPU Memory", f"{gpu_mem:.1f}%")
        else:
            st.warning("⚠️ GPU情報が取得できません。GPUが利用できないか、ドライバーが正しくインストールされていません。")
        
        # CPU/メモリ情報
        st.markdown('<h3>🖥️ CPU & MEMORY</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_usage = metrics.get('cpu_usage', 0.0)
            fig_cpu = create_gauge_chart(
                cpu_usage,
                "CPU Usage",
                max_value=100.0,
                warning_threshold=80.0
            )
            st.plotly_chart(fig_cpu, use_container_width=True)
            st.metric("CPU Usage", f"{cpu_usage:.1f}%")
        
        with col2:
            mem_usage = metrics.get('memory_usage', 0.0)
            fig_mem = create_gauge_chart(
                mem_usage,
                "Memory Usage",
                max_value=100.0,
                warning_threshold=85.0
            )
            st.plotly_chart(fig_mem, use_container_width=True)
            st.metric("Memory Usage", f"{mem_usage:.1f}%")
        
        # 最新メトリクスサマリー
        st.markdown('<h3>📊 METRICS SUMMARY</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("GPU Temp", f"{metrics.get('gpu_temperature', 0.0):.1f}°C")
        with col2:
            st.metric("GPU Usage", f"{metrics.get('gpu_usage', 0.0):.1f}%")
        with col3:
            st.metric("GPU Memory", f"{metrics.get('gpu_memory_usage', 0.0):.1f}%")
        with col4:
            st.metric("CPU Usage", f"{metrics.get('cpu_usage', 0.0):.1f}%")
        with col5:
            st.metric("Memory", f"{metrics.get('memory_usage', 0.0):.1f}%")
    
    with tab2:
        # 時系列チャート
        st.markdown('<h2>📈 TIME SERIES CHARTS</h2>', unsafe_allow_html=True)
        
        if len(st.session_state.metrics_history) > 0:
            # GPUチャート
            if metrics.get('gpu_available', False):
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_gpu_ts = create_timeseries_chart(
                        st.session_state.metrics_history,
                        'gpu_usage',
                        'GPU Usage Over Time',
                        '#00ff41'
                    )
                    st.plotly_chart(fig_gpu_ts, use_container_width=True)
                
                with col2:
                    fig_gpu_mem_ts = create_timeseries_chart(
                        st.session_state.metrics_history,
                        'gpu_memory_usage',
                        'GPU Memory Over Time',
                        '#00ffff'
                    )
                    st.plotly_chart(fig_gpu_mem_ts, use_container_width=True)
                
                # GPU温度チャート
                fig_gpu_temp_ts = create_timeseries_chart(
                    st.session_state.metrics_history,
                    'gpu_temperature',
                    'GPU Temperature Over Time',
                    '#ff0040'
                )
                st.plotly_chart(fig_gpu_temp_ts, use_container_width=True)
            
            # CPU/メモリチャート
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cpu_ts = create_timeseries_chart(
                    st.session_state.metrics_history,
                    'cpu_usage',
                    'CPU Usage Over Time',
                    '#ffaa00'
                )
                st.plotly_chart(fig_cpu_ts, use_container_width=True)
            
            with col2:
                fig_mem_ts = create_timeseries_chart(
                    st.session_state.metrics_history,
                    'memory_usage',
                    'Memory Usage Over Time',
                    '#ff00ff'
                )
                st.plotly_chart(fig_mem_ts, use_container_width=True)
        else:
            st.info("メトリクス履歴がありません。しばらくお待ちください...")
    
    with tab3:
        # ログストリーミング
        st.markdown('<h2>📝 LOG STREAMING</h2>', unsafe_allow_html=True)
        
        if log_paths:
            for log_path in log_paths:
                if log_path.exists():
                    st.markdown(f'<h3>📄 {log_path.name}</h3>', unsafe_allow_html=True)
                    
                    log_lines = load_log_file(log_path, max_lines=50)
                    
                    if log_lines:
                        log_html = '<div class="log-container">'
                        for line in log_lines:
                            formatted_line = format_log_line(line)
                            if formatted_line:
                                log_html += f'<div class="log-line">{formatted_line}</div>'
                        log_html += '</div>'
                        
                        st.markdown(log_html, unsafe_allow_html=True)
                        
                        # ログファイルの更新時刻
                        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
                        st.caption(f"Last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.info(f"No log entries found in {log_path.name}")
                else:
                    st.warning(f"⚠️ Log file not found: {log_path}")
        else:
            st.info("ログファイルが設定されていません。サイドバーでログファイルパスを設定してください。")
    
    # 自動更新
    if auto_refresh:
        # セッション状態を更新
        st.session_state.update_count += 1
        st.session_state.last_update_time = time.time()
        
        # 更新状態を表示
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span style="font-family: 'Orbitron', monospace; color: #00ff00; text-shadow: 0 0 10px #00ff00;">
                    ⚡ UPDATING... (更新回数: {st.session_state.update_count})
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # 更新間隔待機
        time.sleep(refresh_interval)
        
        # 再実行
        st.rerun()


if __name__ == "__main__":
    main()

