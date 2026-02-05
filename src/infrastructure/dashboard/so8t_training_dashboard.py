#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T再学習進捗Streamlitダッシュボード（サイバーパンク風）

リアルタイムで学習進捗、システムメトリクス、学習曲線を可視化
"""

import sys
import yaml
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time

# Streamlitの警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# ロギング設定
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# StreamlitのScriptRunContext警告を抑制
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.dashboard_utils import (
    load_progress_logs,
    load_session_info,
    load_checkpoint_info,
    calculate_progress,
    estimate_remaining_time,
    get_latest_session,
    get_elapsed_time,
    parse_training_log,
    load_training_session,
    get_system_metrics
)

# サイバーパンク風CSS
CYBERPUNK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    /* メイン背景 */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #0a0a0a 100%);
        background-attachment: fixed;
    }
    
    /* ヘッダー */
    h1, h2, h3 {
        font-family: 'Orbitron', monospace;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        letter-spacing: 2px;
    }
    
    /* メトリクス */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace;
        color: #00ff00;
        text-shadow: 0 0 5px #00ff00;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Orbitron', monospace;
        color: #00ffff;
        text-shadow: 0 0 5px #00ffff;
    }
    
    /* サイドバー */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a0033 0%, #0a0a0a 100%);
        border-right: 2px solid #00ffff;
        box-shadow: 0 0 20px #00ffff;
    }
    
    /* カード */
    .stCard {
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid #00ffff;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        padding: 20px;
    }
    
    /* プログレスバー */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ffff 0%, #00ff00 50%, #ff00ff 100%);
        box-shadow: 0 0 10px #00ffff;
    }
    
    /* ボタン */
    .stButton > button {
        background: linear-gradient(135deg, #00ffff 0%, #00ff00 100%);
        color: #000;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        border: 2px solid #00ffff;
        border-radius: 5px;
        box-shadow: 0 0 10px #00ffff;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px #00ffff;
        transform: scale(1.05);
    }
    
    /* テキストエリア */
    .stTextArea > div > div > textarea {
        background: rgba(0, 0, 0, 0.8);
        color: #00ff00;
        font-family: 'Courier New', monospace;
        border: 1px solid #00ffff;
        border-radius: 5px;
    }
    
    /* タブ */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.5);
        border-bottom: 2px solid #00ffff;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00ff00;
        text-shadow: 0 0 5px #00ff00;
    }
    
    /* グリッチエフェクト */
    @keyframes glitch {
        0%, 100% { transform: translate(0); }
        20% { transform: translate(-2px, 2px); }
        40% { transform: translate(-2px, -2px); }
        60% { transform: translate(2px, 2px); }
        80% { transform: translate(2px, -2px); }
    }
    
    .glitch {
        animation: glitch 0.3s infinite;
    }
    
    /* パルスエフェクト */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
"""

# ページ設定
st.set_page_config(
    page_title="SO8T Cyber Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSSを適用
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# 設定読み込み
@st.cache_data
def load_config() -> Dict[str, Any]:
    """
    設定ファイルを読み込み
    
    Returns:
        設定辞書（デフォルト値を含む）
    """
    config_path = PROJECT_ROOT / "configs" / "dashboard_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None:
                    raise ValueError("Config file is empty")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    
    # デフォルト設定を返す
    return {
        'checkpoint_base_dir': 'D:/webdataset/checkpoints/training',
        'refresh_interval': 5,
        'port': 8501,
        'gpu_temp_warning': 75
    }


def create_gauge_chart(
    value: float, 
    title: str, 
    max_value: float = 100.0, 
    warning_threshold: Optional[float] = None
) -> go.Figure:
    """
    サイバーパンク風ゲージチャートを作成
    
    Args:
        value: 表示する値
        title: チャートタイトル
        max_value: 最大値（デフォルト: 100.0）
        warning_threshold: 警告閾値（オプション）
    
    Returns:
        Plotly Figure オブジェクト
    
    Raises:
        ValueError: 無効な値が渡された場合
    """
    # 値の検証
    if not isinstance(value, (int, float)):
        raise ValueError(f"Value must be numeric, got {type(value)}")
    if not isinstance(max_value, (int, float)) or max_value <= 0:
        raise ValueError(f"max_value must be positive numeric, got {max_value}")
    if value < 0 or value > max_value:
        logger.warning(f"Value {value} is out of range [0, {max_value}], clamping")
        value = max(0.0, min(float(value), float(max_value)))
    
    # サイバーパンク風の色決定
    try:
        if warning_threshold is not None and isinstance(warning_threshold, (int, float)) and value >= warning_threshold:
            color = '#ff0080'  # マゼンタ
            bg_color = 'rgba(255, 0, 128, 0.2)'
        elif value >= max_value * 0.8:
            color = '#ffff00'  # イエロー
            bg_color = 'rgba(255, 255, 0, 0.2)'
        else:
            color = '#00ff00'  # グリーン
            bg_color = 'rgba(0, 255, 0, 0.2)'
    except Exception as e:
        logger.error(f"Error determining gauge color: {e}")
        color = '#00ff00'
        bg_color = 'rgba(0, 255, 0, 0.2)'
    
    try:
        fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<span style='font-family: Orbitron; color: #00ffff; font-size: 16px;'>{title}</span>",
            'font': {'size': 16}
        },
        number={
            'font': {'family': 'Orbitron', 'size': 24, 'color': color},
            'valueformat': '.1f',
            'suffix': '%'
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
                'value': warning_threshold if warning_threshold else max_value
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
        # エラー時は空のFigureを返す
        fig = go.Figure()
        fig.update_layout(
            height=220,
            title={'text': f'Error: {str(e)}'},
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        return fig


def main() -> None:
    """
    メイン関数
    
    Raises:
        SystemExit: 致命的なエラーが発生した場合
    """
    try:
        # サイバーパンク風タイトル
        st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-family: 'Orbitron', monospace; color: #00ffff; text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff; letter-spacing: 5px; margin: 0;">
            ⚡ SO8T CYBER DASHBOARD ⚡
        </h1>
        <p style="font-family: 'Orbitron', monospace; color: #00ff00; text-shadow: 0 0 10px #00ff00; letter-spacing: 3px; margin-top: 10px;">
            REAL-TIME TRAINING MONITORING SYSTEM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
        # 設定読み込み
        try:
            config = load_config()
            if not isinstance(config, dict):
                raise ValueError("Config must be a dictionary")
            
            checkpoint_base_str: str = config.get('checkpoint_base_dir', 'D:/webdataset/checkpoints/training')
            if not isinstance(checkpoint_base_str, str):
                raise ValueError("checkpoint_base_dir must be a string")
            checkpoint_base = Path(checkpoint_base_str)
            
            refresh_interval: int = int(config.get('refresh_interval', 5))
            if refresh_interval < 1:
                refresh_interval = 1
                logger.warning("refresh_interval must be >= 1, setting to 1")
            
            gpu_temp_warning: float = float(config.get('gpu_temp_warning', 75))
            if gpu_temp_warning < 0:
                gpu_temp_warning = 75.0
                logger.warning("gpu_temp_warning must be >= 0, setting to 75")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            st.error(f"設定の読み込みに失敗しました: {e}")
            checkpoint_base = Path('D:/webdataset/checkpoints/training')
            refresh_interval = 5
            gpu_temp_warning = 75.0
        
        # サイドバー
        with st.sidebar:
            st.markdown("""
        <div style="font-family: 'Orbitron', monospace; color: #00ffff; text-shadow: 0 0 10px #00ffff;">
            <h2>⚙️ CONTROL PANEL</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 更新間隔設定（リアルタイム重視）
        refresh_interval = st.slider("更新間隔（秒）", 1, 10, 2, help="リアルタイム更新のため1-2秒推奨")
        
        # 手動更新ボタン
        if st.button("🔄 FORCE REFRESH", use_container_width=True):
            st.rerun()
        
        # リアルタイムモード
        realtime_mode = st.checkbox("⚡ REALTIME MODE", value=True, help="最高頻度で自動更新")
        if realtime_mode:
            refresh_interval = 1
        
        # セッション選択
        st.header("セッション選択")
        session_dir: Optional[Path] = None
        try:
            if checkpoint_base.exists():
                session_dirs: List[Path] = [d for d in checkpoint_base.iterdir() if d.is_dir()]
                session_names: List[str] = [d.name for d in session_dirs]
            else:
                session_dirs = []
                session_names = []
            
            if session_names:
                selected_session: str = st.selectbox("セッション", session_names, index=0)
                if not isinstance(selected_session, str):
                    raise ValueError("Selected session must be a string")
                session_dir = checkpoint_base / selected_session
            else:
                st.warning("セッションが見つかりません")
                session_dir = None
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            st.error(f"セッションの読み込みに失敗しました: {e}")
            session_dir = None
        
            if session_dir is None or not session_dir.exists():
                st.error(f"セッションディレクトリが見つかりません: {session_dir}")
                return
        
        # データ読み込み
        session_info: Optional[Dict[str, Any]] = None
        try:
            session_info = load_session_info(session_dir)
            # training_session.jsonも試す
            if session_info is None:
                session_info = load_training_session(session_dir)
        except Exception as e:
            logger.error(f"Failed to load session info: {e}")
            session_info = None
        
        progress_logs: List[Dict[str, Any]] = []
        try:
            progress_logs_dir = session_dir / "progress_logs"
            progress_logs = load_progress_logs(progress_logs_dir)
            if not isinstance(progress_logs, list):
                progress_logs = []
        except Exception as e:
            logger.error(f"Failed to load progress logs: {e}")
            progress_logs = []
        
        checkpoint_info: Dict[str, Any] = {}
        try:
            checkpoint_info = load_checkpoint_info(session_dir)
            if not isinstance(checkpoint_info, dict):
                checkpoint_info = {
                    'rolling_checkpoints': [],
                    'final_checkpoint': None,
                    'emergency_checkpoints': [],
                    'hf_checkpoints': [],
                    'time_based_checkpoints': [],
                    'total_count': 0
                }
        except Exception as e:
            logger.error(f"Failed to load checkpoint info: {e}")
            checkpoint_info = {
                'rolling_checkpoints': [],
                'final_checkpoint': None,
                'emergency_checkpoints': [],
                'hf_checkpoints': [],
                'time_based_checkpoints': [],
                'total_count': 0
            }
        
        # ログファイルを解析（最新のログを確実に読み込む）
        log_info: Dict[str, Any] = {}
        try:
            log_file = PROJECT_ROOT / "logs" / "train_borea_phi35_so8t_thinking.log"
            log_info = parse_training_log(log_file)
            if not isinstance(log_info, dict):
                log_info = {}
        except Exception as e:
            logger.error(f"Failed to parse training log: {e}")
            log_info = {}
        
        # ログファイルの更新時刻を表示（サイドバー）
        try:
            log_file_mtime_str = log_info.get('log_file_mtime')
            if log_file_mtime_str and isinstance(log_file_mtime_str, str):
                log_mtime = datetime.fromisoformat(log_file_mtime_str)
                now = datetime.now()
                elapsed = (now - log_mtime).total_seconds()
                if elapsed < 0:
                    elapsed = 0
                elapsed_str = (
                    f"{int(elapsed // 60)}分{int(elapsed % 60)}秒前" 
                    if elapsed < 3600 
                    else f"{int(elapsed // 3600)}時間{int((elapsed % 3600) // 60)}分前"
                )
                log_file_size = log_info.get('log_file_size', 0)
                if not isinstance(log_file_size, (int, float)):
                    log_file_size = 0
                st.sidebar.markdown(f"""
                <div style="font-family: 'Orbitron', monospace; color: #00ff00; font-size: 12px; margin-top: 20px;">
                    📝 ログ最終更新: {elapsed_str}<br>
                    📊 ログサイズ: {log_file_size / 1024 / 1024:.2f} MB
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            logger.warning(f"Failed to display log file info: {e}")
        
        # システムメトリクスを取得
        system_metrics: Dict[str, Any] = {}
        try:
            system_metrics = get_system_metrics()
            if not isinstance(system_metrics, dict):
                system_metrics = {}
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            system_metrics = {}
        
        # データセット読み込み進捗セクション
        st.markdown("""
    <div style="border: 2px solid #00ffff; border-radius: 10px; padding: 15px; background: rgba(0, 0, 0, 0.5); margin: 20px 0;">
        <h2 style="font-family: 'Orbitron', monospace; color: #00ffff; text-shadow: 0 0 10px #00ffff; margin: 0;">
            📦 DATASET LOADING STATUS
        </h2>
    </div>
    """, unsafe_allow_html=True)
        
        dataset_loading = log_info.get('dataset_loading', {})
        status = dataset_loading.get('status', 'not_started')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_icon = {
                'not_started': '⏸️',
                'loading': '🔄',
                'completed': '✅'
            }.get(status, '❓')
            st.metric("ステータス", f"{status_icon} {status}")
        
        with col2:
            progress = dataset_loading.get('progress', 0.0)
            total_lines = dataset_loading.get('total_lines', 0)
            current_line = dataset_loading.get('current_line', 0)
            if total_lines > 0:
                st.metric("進捗", f"{current_line:,}/{total_lines:,} 行", f"{progress*100:.1f}%")
            else:
                st.metric("進捗", "読み込み中...", "0%")
        
        with col3:
            loaded_samples = dataset_loading.get('loaded_samples', 0)
            if loaded_samples > 0:
                st.metric("読み込み済みサンプル", f"{loaded_samples:,}")
            else:
                st.metric("読み込み済みサンプル", "0")
        
        if status == 'loading' or status == 'completed':
            st.progress(progress)
            if dataset_loading.get('message'):
                st.info(dataset_loading['message'])
        
        # セッション情報セクション
        st.markdown("""
        <div style="border: 2px solid #00ffff; border-radius: 10px; padding: 15px; background: rgba(0, 0, 0, 0.5); margin: 20px 0;">
            <h2 style="font-family: 'Orbitron', monospace; color: #00ffff; text-shadow: 0 0 10px #00ffff; margin: 0;">
                📋 SESSION INFORMATION
            </h2>
        </div>
        """, unsafe_allow_html=True)
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
                except Exception:
                    pass
            st.metric("開始時刻", start_time)
        with col4:
            elapsed_time = get_elapsed_time(session_info)
            st.metric("経過時間", elapsed_time if elapsed_time else 'N/A')
        
        # 学習進捗セクション
        st.markdown("""
        <div style="border: 2px solid #00ff00; border-radius: 10px; padding: 15px; background: rgba(0, 0, 0, 0.5); margin: 20px 0;">
            <h2 style="font-family: 'Orbitron', monospace; color: #00ff00; text-shadow: 0 0 10px #00ff00; margin: 0;">
                📈 TRAINING PROGRESS
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # ログファイルから学習進捗情報を取得
        training_info = log_info.get('training', {})
        
        # セッション情報から情報を取得（優先）
        if session_info:
            current_epoch = session_info.get('current_epoch', training_info.get('current_epoch', 0))
            current_step = session_info.get('current_step', training_info.get('current_step', 0))
            total_steps = session_info.get('total_steps', training_info.get('total_steps', 0))
            best_loss = session_info.get('best_loss', float('inf'))
            num_epochs = session_info.get('num_epochs', training_info.get('total_epochs', 3))
        else:
            current_epoch = training_info.get('current_epoch', 0)
            current_step = training_info.get('current_step', 0)
            total_steps = training_info.get('total_steps', 0)
            best_loss = float('inf')
            num_epochs = training_info.get('total_epochs', 3)
        
        # ログファイルから損失値と学習率を取得
        current_loss = training_info.get('loss', 0.0)
        learning_rate = training_info.get('learning_rate', 0.0)
        
        # 進捗ログからも情報を取得（優先）
        if progress_logs and isinstance(progress_logs, list) and len(progress_logs) > 0:
            try:
                latest_log = progress_logs[-1]
                if isinstance(latest_log, dict):
                    loss_val = latest_log.get('loss')
                    if loss_val is not None and isinstance(loss_val, (int, float)):
                        current_loss = float(loss_val)
                    
                    lr_val = latest_log.get('learning_rate')
                    if lr_val is not None and isinstance(lr_val, (int, float)):
                        learning_rate = float(lr_val)
                    
                    step_val = latest_log.get('step')
                    if step_val is not None and isinstance(step_val, int):
                        current_step = int(step_val)
            except Exception as e:
                logger.warning(f"Failed to extract info from progress logs: {e}")
        
        # 進捗率計算
        try:
            progress: float = calculate_progress(current_step, total_steps) if total_steps > 0 else 0.0
            if not isinstance(progress, (int, float)) or progress < 0 or progress > 1:
                progress = 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate progress: {e}")
            progress = 0.0
        
        # 残り時間推定
        remaining_time: Optional[str] = None
        try:
            if progress_logs and isinstance(progress_logs, list):
                remaining_time = estimate_remaining_time(progress_logs, current_step, total_steps)
        except Exception as e:
            logger.warning(f"Failed to estimate remaining time: {e}")
            remaining_time = None
        
        if total_steps > 0 or current_step > 0:
            col1, col2, col3, col4, col5 = st.columns(5)
        
            with col1:
                st.metric("エポック", f"{current_epoch}/{num_epochs}")
            with col2:
                st.metric("ステップ", f"{current_step:,}/{total_steps:,}", f"{progress*100:.1f}%" if total_steps > 0 else None)
            with col3:
                loss_delta = f"Best: {best_loss:.4f}" if best_loss != float('inf') else None
                st.metric("損失値", f"{current_loss:.4f}", loss_delta)
            with col4:
                st.metric("学習率", f"{learning_rate:.2e}" if learning_rate > 0 else "N/A")
            with col5:
                st.metric("推定残り時間", remaining_time if remaining_time else 'N/A')
            
            # 進捗バー
            if total_steps > 0:
                st.progress(progress)
            
            # 学習ステータス
            training_status = training_info.get('status', 'not_started')
            status_icon = {
                'not_started': '⏸️',
                'running': '🔄',
                'completed': '✅'
            }.get(training_status, '❓')
            st.info(f"{status_icon} 学習ステータス: {training_status}")
        else:
            st.info("学習進捗データがありません")
        
        # システムメトリクスセクション
        st.markdown("""
    <div style="border: 2px solid #ff00ff; border-radius: 10px; padding: 15px; background: rgba(0, 0, 0, 0.5); margin: 20px 0;">
        <h2 style="font-family: 'Orbitron', monospace; color: #ff00ff; text-shadow: 0 0 10px #ff00ff; margin: 0;">
            💻 SYSTEM METRICS
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
        # リアルタイムシステムメトリクスを使用（ログから取得できない場合）
        cpu_usage: float = 0.0
        memory_usage: float = 0.0
        gpu_usage: float = 0.0
        gpu_memory_usage: float = 0.0
        gpu_temp: float = 0.0
        
        try:
            if progress_logs and isinstance(progress_logs, list) and len(progress_logs) > 0:
                latest_log = progress_logs[-1]
                if isinstance(latest_log, dict):
                    cpu_usage = float(latest_log.get('cpu_usage', system_metrics.get('cpu_usage', 0.0)) or 0.0)
                    memory_usage = float(latest_log.get('memory_usage', system_metrics.get('memory_usage', 0.0)) or 0.0)
                    gpu_usage = float(latest_log.get('gpu_usage', system_metrics.get('gpu_usage', 0.0)) or 0.0)
                    gpu_memory_usage = float(latest_log.get('gpu_memory_usage', system_metrics.get('gpu_memory_usage', 0.0)) or 0.0)
                    gpu_temp = float(latest_log.get('gpu_temperature', system_metrics.get('gpu_temperature', 0.0)) or 0.0)
                else:
                    cpu_usage = float(system_metrics.get('cpu_usage', 0.0) or 0.0)
                    memory_usage = float(system_metrics.get('memory_usage', 0.0) or 0.0)
                    gpu_usage = float(system_metrics.get('gpu_usage', 0.0) or 0.0)
                    gpu_memory_usage = float(system_metrics.get('gpu_memory_usage', 0.0) or 0.0)
                    gpu_temp = float(system_metrics.get('gpu_temperature', 0.0) or 0.0)
            else:
                cpu_usage = float(system_metrics.get('cpu_usage', 0.0) or 0.0)
                memory_usage = float(system_metrics.get('memory_usage', 0.0) or 0.0)
                gpu_usage = float(system_metrics.get('gpu_usage', 0.0) or 0.0)
                gpu_memory_usage = float(system_metrics.get('gpu_memory_usage', 0.0) or 0.0)
                gpu_temp = float(system_metrics.get('gpu_temperature', 0.0) or 0.0)
        except Exception as e:
            logger.error(f"Failed to extract system metrics: {e}")
            cpu_usage = 0.0
            memory_usage = 0.0
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            gpu_temp = 0.0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            try:
                fig_cpu = create_gauge_chart(cpu_usage, "CPU使用率", max_value=100.0)
                st.plotly_chart(fig_cpu, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to create CPU gauge: {e}")
                st.error(f"CPUゲージの作成に失敗しました: {e}")
        
        with col2:
            try:
                fig_memory = create_gauge_chart(memory_usage, "メモリ使用率", max_value=100.0)
                st.plotly_chart(fig_memory, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to create memory gauge: {e}")
                st.error(f"メモリゲージの作成に失敗しました: {e}")
        
        with col3:
            try:
                gpu_available = system_metrics.get('gpu_available', False)
                if isinstance(gpu_available, bool) and gpu_available:
                    fig_gpu = create_gauge_chart(gpu_usage, "GPU使用率", max_value=100.0)
                    st.plotly_chart(fig_gpu, use_container_width=True)
                else:
                    st.info("GPU情報が取得できません")
            except Exception as e:
                logger.error(f"Failed to create GPU gauge: {e}")
                st.info("GPU情報が取得できません")
        
        with col4:
            try:
                gpu_available = system_metrics.get('gpu_available', False)
                if isinstance(gpu_available, bool) and gpu_available:
                    fig_gpu_mem = create_gauge_chart(gpu_memory_usage, "GPUメモリ使用率", max_value=100.0)
                    st.plotly_chart(fig_gpu_mem, use_container_width=True)
                else:
                    st.info("GPU情報が取得できません")
            except Exception as e:
                logger.error(f"Failed to create GPU memory gauge: {e}")
                st.info("GPU情報が取得できません")
        
        with col5:
            try:
                gpu_available = system_metrics.get('gpu_available', False)
                if isinstance(gpu_available, bool) and gpu_available:
                    fig_gpu_temp = create_gauge_chart(gpu_temp, "GPU温度", max_value=100.0, warning_threshold=gpu_temp_warning)
                    st.plotly_chart(fig_gpu_temp, use_container_width=True)
                else:
                    st.info("GPU情報が取得できません")
            except Exception as e:
                logger.error(f"Failed to create GPU temperature gauge: {e}")
                st.info("GPU情報が取得できません")
        
        # 学習曲線セクション
        st.markdown("""
    <div style="border: 2px solid #00ffff; border-radius: 10px; padding: 15px; background: rgba(0, 0, 0, 0.5); margin: 20px 0;">
        <h2 style="font-family: 'Orbitron', monospace; color: #00ffff; text-shadow: 0 0 10px #00ffff; margin: 0;">
            📊 TRAINING CURVES
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
        if progress_logs and isinstance(progress_logs, list) and len(progress_logs) > 0:
            try:
                # データフレーム作成
                df = pd.DataFrame(progress_logs)
                
                # タイムスタンプをdatetimeに変換
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                    except Exception as e:
                        logger.warning(f"Failed to parse timestamps: {e}")
            except Exception as e:
                logger.error(f"Failed to create DataFrame from progress logs: {e}")
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # 損失値の推移（サイバーパンク風）
                try:
                    if 'loss' in df.columns and 'step' in df.columns:
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            x=df['step'],
                            y=df['loss'],
                            mode='lines',
                            name='Loss',
                            line=dict(color='#00ff00', width=3),
                            fill='tonexty',
                            fillcolor='rgba(0, 255, 0, 0.1)'
                        ))
                        fig_loss.update_layout(
                            title={
                                'text': '<span style="font-family: Orbitron; color: #00ff00; font-size: 18px;">LOSS CURVE</span>',
                                'x': 0.5
                            },
                            xaxis_title='<span style="font-family: Orbitron; color: #00ffff;">STEP</span>',
                            yaxis_title='<span style="font-family: Orbitron; color: #00ffff;">LOSS</span>',
                            height=400,
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0.5)',
                            font={'family': 'Orbitron', 'color': '#00ffff'},
                            xaxis=dict(gridcolor='rgba(0, 255, 255, 0.2)', showgrid=True),
                            yaxis=dict(gridcolor='rgba(0, 255, 255, 0.2)', showgrid=True)
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                    else:
                        st.info("損失値データがありません")
                except Exception as e:
                    logger.error(f"Failed to create loss curve: {e}")
                    st.error(f"損失値グラフの作成に失敗しました: {e}")
            
            with col2:
                # 学習率の推移（サイバーパンク風）
                try:
                    if 'learning_rate' in df.columns and 'step' in df.columns:
                        fig_lr = go.Figure()
                        fig_lr.add_trace(go.Scatter(
                            x=df['step'],
                            y=df['learning_rate'],
                            mode='lines',
                            name='Learning Rate',
                            line=dict(color='#ff00ff', width=3),
                            fill='tonexty',
                            fillcolor='rgba(255, 0, 255, 0.1)'
                        ))
                        fig_lr.update_layout(
                            title={
                                'text': '<span style="font-family: Orbitron; color: #ff00ff; font-size: 18px;">LEARNING RATE CURVE</span>',
                                'x': 0.5
                            },
                            xaxis_title='<span style="font-family: Orbitron; color: #00ffff;">STEP</span>',
                            yaxis_title='<span style="font-family: Orbitron; color: #00ffff;">LEARNING RATE</span>',
                            height=400,
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0.5)',
                            font={'family': 'Orbitron', 'color': '#00ffff'},
                            xaxis=dict(gridcolor='rgba(0, 255, 255, 0.2)', showgrid=True),
                            yaxis=dict(gridcolor='rgba(0, 255, 255, 0.2)', showgrid=True)
                        )
                        st.plotly_chart(fig_lr, use_container_width=True)
                    else:
                        st.info("学習率データがありません")
                except Exception as e:
                    logger.error(f"Failed to create learning rate curve: {e}")
                    st.error(f"学習率グラフの作成に失敗しました: {e}")
            
            # システムメトリクスの推移
            st.subheader("システムメトリクスの推移")
            try:
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
            except Exception as e:
                logger.error(f"Failed to create metrics chart: {e}")
                st.error(f"システムメトリクスグラフの作成に失敗しました: {e}")
        else:
            st.info("学習曲線データがありません")
        
        # ログファイル表示セクション
        st.header("📄 ログファイル")
        
        tab1, tab2, tab3 = st.tabs(["最新ログ", "エラー", "警告"])
        
        with tab1:
            try:
                latest_logs = log_info.get('latest_logs', [])
                if isinstance(latest_logs, list) and latest_logs:
                    log_lines = '\n'.join(str(line) for line in latest_logs[-50:])  # 最新50行
                    st.text_area("最新ログ（最新50行）", log_lines, height=400, key="latest_logs")
                else:
                    st.info("ログデータがありません")
            except Exception as e:
                logger.error(f"Failed to display latest logs: {e}")
                st.error(f"ログの表示に失敗しました: {e}")
        
        with tab2:
            try:
                errors = log_info.get('errors', [])
                if isinstance(errors, list) and errors:
                    error_text = '\n'.join(str(err) for err in errors[:20])  # 最新20件
                    st.text_area("エラーログ", error_text, height=400, key="error_logs")
                else:
                    st.success("エラーはありません")
            except Exception as e:
                logger.error(f"Failed to display errors: {e}")
                st.error(f"エラーログの表示に失敗しました: {e}")
        
        with tab3:
            try:
                warnings = log_info.get('warnings', [])
                if isinstance(warnings, list) and warnings:
                    warning_text = '\n'.join(str(warn) for warn in warnings[:20])  # 最新20件
                    st.text_area("警告ログ", warning_text, height=400, key="warning_logs")
                else:
                    st.info("警告はありません")
            except Exception as e:
                logger.error(f"Failed to display warnings: {e}")
                st.error(f"警告ログの表示に失敗しました: {e}")
        
        # チェックポイント情報セクション
        st.header("💾 チェックポイント情報")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("最新チェックポイント")
            latest_checkpoint: Optional[Path] = None
            
            try:
                # HuggingFace形式のチェックポイントを優先
                hf_checkpoints = checkpoint_info.get('hf_checkpoints', [])
                if isinstance(hf_checkpoints, list) and hf_checkpoints:
                    latest_checkpoint = Path(str(hf_checkpoints[0]))
                elif checkpoint_info.get('rolling_checkpoints'):
                    rolling = checkpoint_info['rolling_checkpoints']
                    if isinstance(rolling, list) and rolling:
                        latest_checkpoint = Path(str(rolling[0]))
                elif checkpoint_info.get('time_based_checkpoints'):
                    time_based = checkpoint_info['time_based_checkpoints']
                    if isinstance(time_based, list) and time_based:
                        latest_checkpoint = Path(str(time_based[0]))
                elif checkpoint_info.get('final_checkpoint'):
                    final = checkpoint_info['final_checkpoint']
                    if final:
                        latest_checkpoint = Path(str(final))
                
                if latest_checkpoint and latest_checkpoint.exists():
                    st.write(f"**ファイル名**: {latest_checkpoint.name}")
                    st.write(f"**パス**: {latest_checkpoint}")
                    try:
                        mtime = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
                        st.write(f"**更新時刻**: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        logger.warning(f"Failed to get checkpoint mtime: {e}")
                else:
                    st.info("チェックポイントが見つかりません")
            except Exception as e:
                logger.error(f"Failed to load checkpoint info: {e}")
                st.error(f"チェックポイント情報の読み込みに失敗しました: {e}")
        
        with col2:
            st.subheader("チェックポイント統計")
            try:
                hf_count = len(checkpoint_info.get('hf_checkpoints', [])) if isinstance(checkpoint_info.get('hf_checkpoints'), list) else 0
                time_based_count = len(checkpoint_info.get('time_based_checkpoints', [])) if isinstance(checkpoint_info.get('time_based_checkpoints'), list) else 0
                rolling_count = len(checkpoint_info.get('rolling_checkpoints', [])) if isinstance(checkpoint_info.get('rolling_checkpoints'), list) else 0
                emergency_count = len(checkpoint_info.get('emergency_checkpoints', [])) if isinstance(checkpoint_info.get('emergency_checkpoints'), list) else 0
                total_count = checkpoint_info.get('total_count', 0) if isinstance(checkpoint_info.get('total_count'), int) else 0
                final_exists = bool(checkpoint_info.get('final_checkpoint'))
                
                st.metric("HuggingFace形式", hf_count)
                st.metric("時間ベース", time_based_count)
                st.metric("ローリングストック", f"{rolling_count}/5")
                st.metric("最終チェックポイント", "あり" if final_exists else "なし")
                st.metric("緊急チェックポイント", emergency_count)
                st.metric("総チェックポイント数", total_count)
            except Exception as e:
                logger.error(f"Failed to display checkpoint statistics: {e}")
                st.error(f"チェックポイント統計の表示に失敗しました: {e}")
        
        # リアルタイム自動更新
        auto_refresh = st.checkbox("🔄 AUTO REFRESH", value=True, key="auto_refresh")
        
        # セッション状態で更新回数を追跡
        if 'update_count' not in st.session_state:
            st.session_state.update_count = 0
        if 'last_update_time' not in st.session_state:
            st.session_state.last_update_time = time.time()
        
        # 更新状態を表示
        current_time = time.time()
        elapsed_since_update = current_time - st.session_state.last_update_time
        st.session_state.update_count += 1
        
        # 更新情報を表示
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**更新回数**: {st.session_state.update_count}")
            st.markdown(f"**最終更新**: {datetime.now().strftime('%H:%M:%S')}")
            if log_file.exists():
                log_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                st.markdown(f"**ログ更新**: {log_mtime.strftime('%H:%M:%S')}")
                time_since_log_update = (current_time - log_file.stat().st_mtime)
                if time_since_log_update < 60:
                    st.markdown(f"**ログ更新**: {int(time_since_log_update)}秒前", help="ログファイルが最近更新されました")
                else:
                    st.markdown(f"**ログ更新**: {int(time_since_log_update/60)}分前", help="ログファイルの更新が止まっている可能性があります")
        
        if realtime_mode or auto_refresh:
            try:
                # ログファイルの変更を検知
                log_file_mtime: float = 0.0
                if log_file.exists():
                    try:
                        log_file_mtime = log_file.stat().st_mtime
                    except Exception as e:
                        logger.warning(f"Failed to get log file mtime: {e}")
                
                current_time = time.time()
                
                # ログファイルが更新されている場合は即座に更新
                try:
                    if current_time - log_file_mtime < refresh_interval:
                        time.sleep(0.5)  # より短い待機
                    else:
                        time.sleep(refresh_interval)
                except Exception as e:
                    logger.warning(f"Failed to sleep: {e}")
                    time.sleep(1)  # フォールバック
                
                # プログレスバーで更新状態を表示
                try:
                    placeholder = st.empty()
                    with placeholder.container():
                        update_count_str = str(st.session_state.get('update_count', 0))
                        st.markdown(f"""
                        <div style="text-align: center; padding: 10px;">
                            <span style="font-family: 'Orbitron', monospace; color: #00ff00; text-shadow: 0 0 10px #00ff00;">
                                ⚡ UPDATING... (更新回数: {update_count_str})
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    logger.warning(f"Failed to display update status: {e}")
                
                # セッション状態を更新
                try:
                    st.session_state.last_update_time = time.time()
                except Exception as e:
                    logger.warning(f"Failed to update session state: {e}")
                
                # 強制的に再実行
                try:
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to rerun: {e}")
                    st.error(f"更新エラー: {e}")
                    # エラーが発生した場合でも再試行
                    time.sleep(1)
                    try:
                        st.rerun()
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Failed in auto-refresh loop: {e}")
                st.error(f"自動更新でエラーが発生しました: {e}")
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        st.error(f"致命的なエラーが発生しました: {e}")
        st.exception(e)


if __name__ == '__main__':
    main()

