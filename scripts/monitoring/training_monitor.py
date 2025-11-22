#!/usr/bin/env python3
"""
SO8T/thinking トレーニングリアルタイム監視ダッシュボード
トレーニングの進捗、GPU温度使用率などを監視

特徴:
- リアルタイムトレーニング進捗表示
- GPU/CPU使用率と温度監視
- 損失関数と学習率の推移グラフ
- 推定残り時間の計算
- トレーニングログのストリーミング
"""

import sys
import os
import time
import json
import subprocess
import psutil
import GPUtil
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue
import re

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TrainingMonitor:
    """トレーニング監視クラス"""

    def __init__(self):
        self.training_start_time = None
        self.max_steps = 500
        self.log_queue = queue.Queue()
        self.metrics_history = []
        self.current_step = 0
        self.current_loss = 0.0
        self.current_lr = 0.0

    def get_gpu_info(self):
        """GPU情報を取得"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 最初のGPUを使用
                return {
                    'usage': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature
                }
        except Exception as e:
            print(f"GPU info error: {e}")

        return {
            'usage': 0,
            'memory_used': 0,
            'memory_total': 0,
            'memory_percent': 0,
            'temperature': 0
        }

    def get_system_info(self):
        """システム情報を取得"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }

    def parse_training_log(self, log_line):
        """トレーニングログを解析"""
        # ログから情報を抽出する正規表現パターン
        patterns = {
            'step': r'Step (\d+)/(\d+)',
            'loss': r'loss[:\s]+([0-9.]+)',
            'learning_rate': r'learning_rate[:\s]+([0-9.e-]+)',
            'epoch': r'epoch[:\s]+([0-9.]+)',
            'gpu_memory': r'GPU memory[:\s]+([0-9.]+)GB',
        }

        info = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                if key in ['step', 'epoch']:
                    info[key] = int(float(match.group(1)))
                else:
                    info[key] = float(match.group(1))

        return info

    def monitor_training_process(self, process_pid):
        """トレーニングプロセスを監視"""
        try:
            process = psutil.Process(process_pid)
            self.training_start_time = datetime.fromtimestamp(process.create_time())

            while True:
                if not process.is_running():
                    break

                # GPUとシステム情報を収集
                gpu_info = self.get_gpu_info()
                system_info = self.get_system_info()

                # タイムスタンプ
                timestamp = datetime.now()

                # メトリクスを記録
                metrics = {
                    'timestamp': timestamp.isoformat(),
                    'gpu_usage': gpu_info['usage'],
                    'gpu_memory_used': gpu_info['memory_used'],
                    'gpu_memory_total': gpu_info['memory_total'],
                    'gpu_memory_percent': gpu_info['memory_percent'],
                    'gpu_temperature': gpu_info['temperature'],
                    'cpu_percent': system_info['cpu_percent'],
                    'memory_percent': system_info['memory_percent'],
                    'memory_used_gb': system_info['memory_used_gb'],
                    'memory_total_gb': system_info['memory_total_gb'],
                    'training_step': self.current_step,
                    'training_loss': self.current_loss,
                    'learning_rate': self.current_lr
                }

                self.metrics_history.append(metrics)

                # 履歴を最新100件に制限
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]

                time.sleep(5)  # 5秒間隔で更新

        except Exception as e:
            print(f"Monitoring error: {e}")

    def estimate_remaining_time(self):
        """残り時間を推定"""
        if not self.training_start_time or self.current_step == 0:
            return "計算中..."

        elapsed_time = datetime.now() - self.training_start_time
        elapsed_seconds = elapsed_time.total_seconds()

        if self.current_step > 0:
            avg_time_per_step = elapsed_seconds / self.current_step
            remaining_steps = self.max_steps - self.current_step
            remaining_seconds = remaining_steps * avg_time_per_step

            remaining_time = timedelta(seconds=int(remaining_seconds))
            return f"{remaining_time.days}日 {remaining_time.seconds//3600}時間 {(remaining_time.seconds//60)%60}分"

        return "計算中..."

    def get_training_progress(self):
        """トレーニング進捗を取得"""
        if self.max_steps == 0:
            return 0.0

        return min(100.0, (self.current_step / self.max_steps) * 100)


# Streamlit ダッシュボード
def create_progress_gauge(value, max_value, title, color='#00ff41'):
    """進捗ゲージを作成"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': color, 'size': 16}},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': color},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'rgba(0, 0, 0, 0.3)'},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': 'rgba(0, 255, 65, 0.2)'},
                {'range': [max_value * 0.8, max_value], 'color': 'rgba(255, 0, 64, 0.2)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': color, 'family': 'Courier New'},
        height=200
    )

    return fig


def create_metrics_chart(metrics_history):
    """メトリクスチャートを作成"""
    if not metrics_history:
        return None

    # 最新50件を使用
    data = metrics_history[-50:]

    df = pd.DataFrame(data)

    # サブプロット作成
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('GPU使用率', 'GPUメモリ使用率', 'GPU温度', 'CPU使用率', 'システムメモリ', 'トレーニング損失'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': True}]]
    )

    # GPU使用率
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gpu_usage'], mode='lines',
                  name='GPU Usage (%)', line=dict(color='#00ff41')),
        row=1, col=1
    )

    # GPUメモリ使用率
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gpu_memory_percent'], mode='lines',
                  name='GPU Memory (%)', line=dict(color='#00ffff')),
        row=1, col=2
    )

    # GPU温度
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gpu_temperature'], mode='lines',
                  name='GPU Temp (°C)', line=dict(color='#ffaa00')),
        row=2, col=1
    )

    # CPU使用率
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_percent'], mode='lines',
                  name='CPU Usage (%)', line=dict(color='#ff0040')),
        row=2, col=2
    )

    # システムメモリ
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_percent'], mode='lines',
                  name='Memory (%)', line=dict(color='#aa00ff')),
        row=3, col=1
    )

    # トレーニング損失 (右軸)
    if 'training_loss' in df.columns and df['training_loss'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['training_loss'], mode='lines',
                      name='Training Loss', line=dict(color='#ffffff')),
            row=3, col=2, secondary_y=True
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#00ff41', 'family': 'Courier New'},
        height=600,
        showlegend=False
    )

    # 軸ラベルの色を設定
    fig.update_xaxes(showticklabels=False)  # X軸ラベルを非表示
    fig.update_yaxes(tickcolor='#00ff41', tickfont=dict(color='#00ff41'))

    return fig


def main():
    st.set_page_config(
        page_title="SO8T/thinking Training Monitor",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # サイバーパンク風CSS
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #0a0a0a 100%);
            color: #00ff41;
            font-family: 'Courier New', monospace;
        }

        .main .block-container {
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff41;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
        }

        h1, h2, h3 {
            color: #00ff41 !important;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
            font-family: 'Courier New', monospace !important;
        }

        .stMetric {
            background: rgba(0, 255, 65, 0.1);
            border: 1px solid #00ff41;
            border-radius: 5px;
            padding: 1rem;
        }

        .metric-label {
            color: #00ff41 !important;
        }

        .metric-value {
            color: #00ff41 !important;
            text-shadow: 0 0 5px rgba(0, 255, 65, 0.5);
        }

        .progress-bar {
            background: linear-gradient(90deg, #00ff41 0%, #00ffff 100%);
        }

        .cyber-border {
            border: 2px solid #00ff41;
            border-radius: 10px;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.5);
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.3);
            margin: 1rem 0;
        }

        .status-running {
            color: #00ff41;
            animation: pulse 2s infinite;
        }

        .status-completed {
            color: #00ff41;
        }

        .status-error {
            color: #ff0040;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .log-container {
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00ff41;
            border-radius: 5px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            color: #00ff41;
            max-height: 400px;
            overflow-y: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="glitch-text">🚀 SO8T/thinking トレーニング監視システム</h1>', unsafe_allow_html=True)

    # トレーニングプロセスIDを取得
    training_pid = st.sidebar.number_input(
        "トレーニングプロセスID",
        value=15680,
        help="PythonトレーニングプロセスのPIDを入力"
    )

    # 監視インスタンス作成
    if 'monitor' not in st.session_state:
        st.session_state.monitor = TrainingMonitor()
        st.session_state.monitor.max_steps = 500

        # 監視スレッドを開始
        monitor_thread = threading.Thread(
            target=st.session_state.monitor.monitor_training_process,
            args=(training_pid,),
            daemon=True
        )
        monitor_thread.start()

    monitor = st.session_state.monitor

    # サイドバー設定
    with st.sidebar:
        st.markdown('<h2>⚙️ コントロールパネル</h2>', unsafe_allow_html=True)

        # 更新間隔
        refresh_interval = st.slider("更新間隔（秒）", 1, 10, 3)

        # 手動更新
        if st.button("🔄 更新", use_container_width=True):
            st.rerun()

        # 自動更新
        auto_refresh = st.checkbox("🔄 自動更新", value=True)

        st.markdown("---")

        # トレーニング情報
        st.markdown('<h3>📊 トレーニング情報</h3>', unsafe_allow_html=True)
        st.write(f"**プロセスID**: {training_pid}")
        st.write(f"**最大ステップ**: {monitor.max_steps}")

        # 現在のメトリクス表示
        if monitor.metrics_history:
            latest = monitor.metrics_history[-1]
            st.markdown('<h4>📈 最新メトリクス</h4>', unsafe_allow_html=True)
            st.write(f"GPU使用率: {latest['gpu_usage']:.1f}%")
            st.write(f"GPU温度: {latest['gpu_temperature']}°C")
            st.write(f"GPUメモリ: {latest['gpu_memory_percent']:.1f}%")
            st.write(f"CPU使用率: {latest['cpu_percent']:.1f}%")

    # メインコンテンツ
    col1, col2, col3, col4 = st.columns(4)

    # リアルタイムメトリクス
    with col1:
        st.markdown('<div class="cyber-border">', unsafe_allow_html=True)
        st.markdown('<h3>🎯 トレーニング進捗</h3>', unsafe_allow_html=True)

        progress = monitor.get_training_progress()
        st.progress(progress / 100)

        st.metric("進捗率", f"{progress:.1f}%")
        st.metric("現在のステップ", monitor.current_step)
        st.metric("総ステップ", monitor.max_steps)

        remaining_time = monitor.estimate_remaining_time()
        st.metric("推定残り時間", remaining_time)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="cyber-border">', unsafe_allow_html=True)
        st.markdown('<h3>🔥 GPU ステータス</h3>', unsafe_allow_html=True)

        if monitor.metrics_history:
            latest = monitor.metrics_history[-1]

            # GPU使用率ゲージ
            fig = create_progress_gauge(latest['gpu_usage'], 100, "GPU使用率")
            st.plotly_chart(fig, use_container_width=True)

            st.metric("GPU温度", f"{latest['gpu_temperature']}°C")
            st.metric("GPUメモリ", f"{latest['gpu_memory_percent']:.1f}%")
        else:
            st.info("メトリクスデータを待機中...")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="cyber-border">', unsafe_allow_html=True)
        st.markdown('<h3>💻 システムリソース</h3>', unsafe_allow_html=True)

        if monitor.metrics_history:
            latest = monitor.metrics_history[-1]

            col3_1, col3_2 = st.columns(2)
            with col3_1:
                st.metric("CPU使用率", f"{latest['cpu_percent']:.1f}%")
            with col3_2:
                st.metric("メモリ使用率", f"{latest['memory_percent']:.1f}%")

            st.metric("メモリ使用量", f"{latest['memory_used_gb']:.1f}GB / {latest['memory_total_gb']:.1f}GB")
        else:
            st.info("メトリクスデータを待機中...")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="cyber-border">', unsafe_allow_html=True)
        st.markdown('<h3>📈 トレーニングメトリクス</h3>', unsafe_allow_html=True)

        col4_1, col4_2 = st.columns(2)
        with col4_1:
            st.metric("現在の損失", f"{monitor.current_loss:.4f}")
        with col4_2:
            st.metric("学習率", f"{monitor.current_lr:.6f}")

        if monitor.training_start_time:
            elapsed = datetime.now() - monitor.training_start_time
            st.metric("経過時間", f"{elapsed.seconds//3600}時間 {(elapsed.seconds//60)%60}分")
        st.markdown('</div>', unsafe_allow_html=True)

    # メトリクスチャート
    st.markdown('<h2>📊 リアルタイムチャート</h2>', unsafe_allow_html=True)

    if monitor.metrics_history:
        fig = create_metrics_chart(monitor.metrics_history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("メトリクスデータを収集中...")

    # ログセクション
    st.markdown('<h2>📝 トレーニングログ</h2>', unsafe_allow_html=True)

    # ログ表示エリア
    log_placeholder = st.empty()

    # ログファイルの読み込みを試行
    log_files = [
        "D:/webdataset/pipeline_logs/master_automated_pipeline.log",
        "logs/training.log",
        "so8t-mmllm/logs/training.log"
    ]

    log_content = ""
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    log_content = ''.join(lines[-50:])  # 最新50行
                break
            except Exception:
                continue

    if log_content:
        st.code(log_content, language='text')
    else:
        st.info("ログファイルが見つかりません。トレーニングが開始されるまでお待ちください。")

    # トレーニングプロセスステータス
    st.markdown('<h2>🔧 プロセスステータス</h2>', unsafe_allow_html=True)

    try:
        process = psutil.Process(training_pid)
        if process.is_running():
            status = "実行中 🟢"
            status_class = "status-running"
        else:
            status = "停止済み 🔴"
            status_class = "status-error"
    except:
        status = "プロセスが見つかりません 🔴"
        status_class = "status-error"

    st.markdown(f'<p class="{status_class}"><strong>ステータス: {status}</strong></p>', unsafe_allow_html=True)

    if monitor.training_start_time:
        st.write(f"開始時刻: {monitor.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # フッター
    st.markdown("---")
    st.caption("SO8T/thinking トレーニング監視システム - リアルタイムGPU・CPU・メモリ監視")

    # 自動更新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()

