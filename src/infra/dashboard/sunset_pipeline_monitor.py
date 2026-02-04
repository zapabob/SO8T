#!/usr/bin/env python3
"""
Sunset Pipeline Monitoring Dashboard with Streamlit
サンセットパイプライン監視ダッシュボード（tqdm + logging + Streamlit）

Real-time monitoring of Sunset Pipeline execution with:
- tqdm progress bars
- logging output
- Streamlit dashboard
"""

import sys
import os
import json
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import queue
import re

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("[ERROR] Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN] tqdm not installed. Install with: pip install tqdm")

import logging
from logging.handlers import QueueHandler, QueueListener

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sunset_pipeline.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# チェックポイントディレクトリ
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sunset_pipeline"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ログキュー（Streamlit表示用）
log_queue = queue.Queue()
log_handler = QueueHandler(log_queue)
logger.addHandler(log_handler)

class SunsetPipelineMonitor:
    """サンセットパイプライン監視クラス"""
    
    def __init__(self):
        self.checkpoint_dir = CHECKPOINT_DIR
        self.logs_dir = PROJECT_ROOT / "logs"
        self.results_dir = PROJECT_ROOT / "results" / "sunset_pipeline"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # パイプライン状態
        self.pipeline_state = {
            'status': 'idle',  # idle, running, completed, failed
            'current_phase': None,
            'phase_progress': {},
            'start_time': None,
            'end_time': None,
            'total_elapsed': 0,
            'phases': {
                'data': {'name': 'Data Pipeline', 'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None},
                'agent_test': {'name': 'Agent Capabilities Testing', 'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None},
                'training': {'name': 'Model Training', 'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None},
                'evaluation': {'name': 'Benchmark Evaluation', 'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None},
                'abc': {'name': 'ABC Comparative Testing', 'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None}
            }
        }
        
        # ログバッファ
        self.log_buffer = []
        self.max_log_lines = 1000
    
    def load_checkpoint(self) -> Optional[Dict]:
        """チェックポイントを読み込み"""
        checkpoint_file = self.checkpoint_dir / "pipeline_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        return None
    
    def save_checkpoint(self, state: Dict):
        """チェックポイントを保存"""
        checkpoint_file = self.checkpoint_dir / "pipeline_checkpoint.json"
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_log_tail(self, num_lines: int = 100) -> List[str]:
        """ログファイルの末尾を取得"""
        log_file = self.logs_dir / "sunset_pipeline.log"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return lines[-num_lines:] if len(lines) > num_lines else lines
            except Exception as e:
                logger.error(f"Failed to read log file: {e}")
        return []
    
    def parse_progress_from_log(self, log_lines: List[str]) -> Dict[str, float]:
        """ログから進捗情報を抽出"""
        progress = {}
        
        # tqdm形式の進捗を検出
        tqdm_pattern = r'(\d+)%\|.*?\| (\d+)/(\d+)'
        phase_patterns = {
            'data': r'\[PHASE 1/5\]|Data Pipeline|データパイプライン',
            'agent_test': r'\[PHASE 2/5\]|Agent Capabilities|エージェント能力',
            'training': r'\[PHASE 3/5\]|Model Training|トレーニング',
            'evaluation': r'\[PHASE 4/5\]|Benchmark Evaluation|ベンチマーク',
            'abc': r'\[PHASE 5/5\]|ABC Comparative|ABC比較'
        }
        
        for line in log_lines:
            # フェーズ検出
            for phase_key, pattern in phase_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    if phase_key not in progress:
                        progress[phase_key] = {'detected': True, 'progress': 0}
            
            # 進捗パーセンテージ検出
            match = re.search(tqdm_pattern, line)
            if match:
                percent = int(match.group(1))
                current = int(match.group(2))
                total = int(match.group(3))
                
                # 現在のフェーズに適用
                for phase_key in progress.keys():
                    if progress[phase_key].get('detected'):
                        progress[phase_key]['progress'] = percent
                        progress[phase_key]['current'] = current
                        progress[phase_key]['total'] = total
        
        return progress
    
    def update_state_from_checkpoint(self):
        """チェックポイントから状態を更新"""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            self.pipeline_state.update(checkpoint)
            return True
        return False

def render_pipeline_status(monitor: SunsetPipelineMonitor):
    """パイプライン状態を表示"""
    st.markdown('<h2>📊 Pipeline Status</h2>', unsafe_allow_html=True)
    
    monitor.update_state_from_checkpoint()
    state = monitor.pipeline_state
    
    # 全体ステータス
    status_colors = {
        'idle': 'gray',
        'running': 'blue',
        'completed': 'green',
        'failed': 'red'
    }
    
    status_color = status_colors.get(state.get('status', 'idle'), 'gray')
    st.markdown(f'<div style="background-color: {status_color}; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px;">'
                f'<strong>Status:</strong> {state.get("status", "idle").upper()}</div>', 
                unsafe_allow_html=True)
    
    # フェーズ別進捗
    st.markdown('<h3>Phase Progress</h3>', unsafe_allow_html=True)
    
    phases = state.get('phases', {})
    for phase_key, phase_info in phases.items():
        phase_name = phase_info.get('name', phase_key)
        phase_status = phase_info.get('status', 'pending')
        phase_progress = phase_info.get('progress', 0)
        
        # ステータスに応じた色
        if phase_status == 'completed':
            status_icon = '✅'
            status_color = 'green'
        elif phase_status == 'running':
            status_icon = '🔄'
            status_color = 'blue'
        elif phase_status == 'failed':
            status_icon = '❌'
            status_color = 'red'
        else:
            status_icon = '⏳'
            status_color = 'gray'
        
        st.markdown(f'<h4>{status_icon} {phase_name}</h4>', unsafe_allow_html=True)
        
        # 進捗バー
        st.progress(phase_progress / 100.0 if phase_progress <= 100 else 1.0)
        st.markdown(f'<p>Progress: {phase_progress}% | Status: {phase_status}</p>', unsafe_allow_html=True)
        
        # 時間情報
        if phase_info.get('start_time'):
            st.markdown(f'<p>Started: {phase_info.get("start_time")}</p>', unsafe_allow_html=True)
        if phase_info.get('end_time'):
            st.markdown(f'<p>Completed: {phase_info.get("end_time")}</p>', unsafe_allow_html=True)
        
        st.markdown('---')

def render_logs_view(monitor: SunsetPipelineMonitor):
    """ログ表示"""
    st.markdown('<h2>📝 Logs</h2>', unsafe_allow_html=True)
    
    # ログファイルから最新のログを取得
    log_lines = monitor.get_log_tail(200)
    
    # ログ表示エリア
    log_text = '\n'.join(log_lines) if log_lines else "No logs available"
    
    st.text_area(
        "Pipeline Logs",
        value=log_text,
        height=400,
        disabled=True,
        key="log_display"
    )
    
    # ログレベルフィルター
    log_level = st.selectbox("Filter Log Level", ["ALL", "INFO", "WARNING", "ERROR"], index=0)
    
    if log_level != "ALL":
        filtered_lines = [line for line in log_lines if f"[{log_level}]" in line]
        if filtered_lines:
            st.text_area(
                f"Filtered Logs ({log_level})",
                value='\n'.join(filtered_lines),
                height=200,
                disabled=True,
                key="filtered_log_display"
            )

def render_progress_bars(monitor: SunsetPipelineMonitor):
    """tqdm風進捗バー表示"""
    st.markdown('<h2>📈 Progress Bars</h2>', unsafe_allow_html=True)
    
    monitor.update_state_from_checkpoint()
    state = monitor.pipeline_state
    
    phases = state.get('phases', {})
    for phase_key, phase_info in phases.items():
        phase_name = phase_info.get('name', phase_key)
        phase_progress = phase_info.get('progress', 0)
        phase_status = phase_info.get('status', 'pending')
        
        # tqdm風の表示
        st.markdown(f'<h4>{phase_name}</h4>', unsafe_allow_html=True)
        
        # 進捗バー
        progress_value = phase_progress / 100.0 if phase_progress <= 100 else 1.0
        st.progress(progress_value)
        
        # 詳細情報
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Progress", f"{phase_progress}%")
        with col2:
            st.metric("Status", phase_status)
        with col3:
            if phase_info.get('start_time'):
                elapsed = datetime.now() - datetime.fromisoformat(phase_info['start_time'].replace('Z', '+00:00'))
                st.metric("Elapsed", str(elapsed).split('.')[0])
        
        st.markdown('---')

def render_statistics(monitor: SunsetPipelineMonitor):
    """統計情報表示"""
    st.markdown('<h2>📊 Statistics</h2>', unsafe_allow_html=True)
    
    monitor.update_state_from_checkpoint()
    state = monitor.pipeline_state
    
    # 全体統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_phases = len(state.get('phases', {}))
        completed_phases = sum(1 for p in state.get('phases', {}).values() if p.get('status') == 'completed')
        st.metric("Completed Phases", f"{completed_phases}/{total_phases}")
    
    with col2:
        if state.get('start_time'):
            start_time = datetime.fromisoformat(state['start_time'].replace('Z', '+00:00'))
            elapsed = datetime.now() - start_time
            st.metric("Total Elapsed", str(elapsed).split('.')[0])
        else:
            st.metric("Total Elapsed", "N/A")
    
    with col3:
        running_phases = sum(1 for p in state.get('phases', {}).values() if p.get('status') == 'running')
        st.metric("Running Phases", running_phases)
    
    with col4:
        failed_phases = sum(1 for p in state.get('phases', {}).values() if p.get('status') == 'failed')
        st.metric("Failed Phases", failed_phases)
    
    # フェーズ別統計
    st.markdown('<h3>Phase Statistics</h3>', unsafe_allow_html=True)
    
    phase_data = []
    for phase_key, phase_info in state.get('phases', {}).items():
        phase_data.append({
            'Phase': phase_info.get('name', phase_key),
            'Status': phase_info.get('status', 'pending'),
            'Progress (%)': phase_info.get('progress', 0),
            'Start Time': phase_info.get('start_time', 'N/A'),
            'End Time': phase_info.get('end_time', 'N/A')
        })
    
    if phase_data:
        import pandas as pd
        df = pd.DataFrame(phase_data)
        st.dataframe(df, use_container_width=True)

def main():
    """メイン関数"""
    st.set_page_config(
        page_title="Sunset Pipeline Monitor",
        page_icon="🌅",
        layout="wide"
    )
    
    # タイトル
    st.markdown('<h1>🌅 Sunset Pipeline Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00ffff; font-size: 18px;">Real-time monitoring with tqdm + logging + Streamlit</p>', unsafe_allow_html=True)
    
    # モニター初期化
    monitor = SunsetPipelineMonitor()
    
    # サイドバー
    with st.sidebar:
        st.markdown('<h2>⚙️ Control Panel</h2>', unsafe_allow_html=True)
        
        # 更新間隔設定
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        # 手動更新ボタン
        if st.button("🔄 Force Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # 自動更新チェックボックス
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        
        st.markdown("---")
        
        # パイプライン制御
        st.markdown('<h3>🚀 Pipeline Control</h3>', unsafe_allow_html=True)
        
        if st.button("▶️ Start Pipeline", use_container_width=True):
            # パイプライン実行スクリプトを起動
            pipeline_script = PROJECT_ROOT / "scripts" / "run_sunset_pipeline.py"
            if pipeline_script.exists():
                st.info("Starting pipeline...")
                # バックグラウンドで実行
                subprocess.Popen(
                    [sys.executable, str(pipeline_script), "--phase", "full"],
                    cwd=str(PROJECT_ROOT)
                )
                st.success("Pipeline started!")
            else:
                st.error("Pipeline script not found")
        
        if st.button("⏹️ Stop Pipeline", use_container_width=True):
            st.warning("Stop functionality not implemented yet")
        
        st.markdown("---")
        
        # セッション情報
        st.markdown('<h3>📊 Session Info</h3>', unsafe_allow_html=True)
        checkpoint = monitor.load_checkpoint()
        if checkpoint:
            st.write(f"**Status**: `{checkpoint.get('status', 'N/A')}`")
            st.write(f"**Current Phase**: `{checkpoint.get('current_phase', 'N/A')}`")
            timestamp = checkpoint.get('start_time', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.write(f"**Start Time**: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    st.write(f"**Start Time**: {timestamp}")
    
    # タブ作成
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Pipeline Status",
        "📈 Progress Bars",
        "📝 Logs",
        "📊 Statistics"
    ])
    
    with tab1:
        render_pipeline_status(monitor)
    
    with tab2:
        render_progress_bars(monitor)
    
    with tab3:
        render_logs_view(monitor)
    
    with tab4:
        render_statistics(monitor)
    
    # 自動更新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
