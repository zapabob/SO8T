#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全自動化マスターパイプライン サイバーパンク風監視ダッシュボード

リアルタイムでパイプライン状態、システムメトリクス、ブラウジング風景を可視化
"""

import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import time

import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.WARNING,  # StreamlitではWARNING以上のみ表示
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
</style>
"""

# ページ設定
st.set_page_config(
    page_title="SO8T Pipeline Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS適用
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=5)
def load_checkpoint(checkpoint_dir: Path) -> Optional[Dict]:
    """最新のチェックポイントを読み込み"""
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("*_checkpoint.json"))
    if not checkpoint_files:
        return None
    
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=5)
def load_progress_logs(logs_dir: Path) -> List[Dict]:
    """進捗ログを読み込み"""
    if not logs_dir.exists():
        return []
    
    log_files = sorted(logs_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    logs = []
    
    for log_file in log_files[:10]:  # 最新10件
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                logs.append({
                    'file': log_file.name,
                    'content': content,
                    'timestamp': datetime.fromtimestamp(log_file.stat().st_mtime)
                })
        except Exception as e:
            logger.debug(f"Failed to load log file {log_file}: {e}")
            continue
    
    return logs


@st.cache_data(ttl=5)
def load_resource_metrics(metrics_dir: Path) -> List[Dict]:
    """リソースメトリクスを読み込み"""
    if not metrics_dir.exists():
        return []
    
    metrics_files = sorted(metrics_dir.glob("metrics_history_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not metrics_files:
        return []
    
    # 最新のメトリクスファイルを読み込み
    latest_file = metrics_files[0]
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # メトリクスの形式を確認（リスト形式または辞書形式）
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'metrics' in data:
                return data.get('metrics', [])
            else:
                return []
    except Exception as e:
        logger.debug(f"Failed to load metrics file {latest_file}: {e}")
        return []


@st.cache_data(ttl=5)
def load_pipeline_log(log_file: Path, max_lines: int = 100) -> List[str]:
    """パイプラインログを読み込み"""
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-max_lines:]  # 最新のN行
    except Exception as e:
        logger.debug(f"Failed to load log file {log_file}: {e}")
        return []


@st.cache_data(ttl=5)
def load_browser_screenshots(screenshots_dir: Path) -> List[Dict]:
    """ブラウザスクリーンショットを読み込み"""
    if not screenshots_dir.exists():
        return []
    
    screenshot_files = sorted(
        screenshots_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    screenshots = []
    for screenshot_file in screenshot_files[:5]:  # 最新5件
        try:
            img = Image.open(screenshot_file)
            screenshots.append({
                'file': screenshot_file.name,
                'path': screenshot_file,
                'image': img,
                'timestamp': datetime.fromtimestamp(screenshot_file.stat().st_mtime)
            })
        except Exception:
            continue
    
    return screenshots


def get_phase_status_color(status: str) -> str:
    """フェーズステータスに応じた色を返す"""
    status_colors = {
        'completed': '#00ff41',
        'running': '#00ffff',
        'failed': '#ff0040',
        'pending': '#888888',
        'skipped': '#ffaa00'
    }
    return status_colors.get(status, '#888888')


def create_cyberpunk_gauge(value: float, max_value: float, title: str, color: str = '#00ff41') -> go.Figure:
    """サイバーパンク風ゲージチャートを作成"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'color': color, 'size': 16}},
        delta = {'reference': max_value * 0.8},
        gauge = {
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
        font={'color': color, 'family': 'Orbitron'},
        height=250
    )
    
    return fig


def create_timeline_chart(phase_results: Dict) -> go.Figure:
    """フェーズタイムラインチャートを作成"""
    phases = []
    start_times = []
    durations = []
    colors = []
    
    phase_labels = {
        'phase0_dependencies': 'Phase 0: Dependencies',
        'phase1_web_scraping': 'Phase 1: Web Scraping',
        'phase2_data_cleansing': 'Phase 2: Data Cleansing',
        'phase3_modeling_so8t': 'Phase 3: SO8T Modeling',
        'phase4_integration': 'Phase 4: Integration',
        'phase5_qlora_training': 'Phase 5: QLoRA Training',
        'phase6_evaluation': 'Phase 6: Evaluation',
        'phase7_ab_test': 'Phase 7: A/B Test',
        'phase8_post_processing': 'Phase 8: Post Processing',
        'phase9_japanese_test': 'Phase 9: Japanese Test'
    }
    
    for phase_key, phase_data in phase_results.items():
        status = phase_data.get('status', 'pending')
        phase_label = phase_labels.get(phase_key, phase_key)
        
        # 開始時刻と終了時刻を推定（チェックポイントのタイムスタンプから）
        # 実際の実装では、より詳細なタイムスタンプ情報が必要
        phases.append(phase_label)
        start_times.append(0)  # 簡易実装
        durations.append(1 if status == 'completed' else 0.5)
        
        color = get_phase_status_color(status)
        colors.append(color)
    
    fig = go.Figure(data=go.Bar(
        x=durations,
        y=phases,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{status.upper()}" for status in [p.get('status', 'pending') for p in phase_results.values()]],
        textposition='inside'
    ))
    
    fig.update_layout(
        title='Pipeline Timeline',
        xaxis_title='Duration',
        yaxis_title='Phase',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#00ff41', 'family': 'Orbitron'},
        height=400
    )
    
    return fig


def create_resource_chart(metrics: List[Dict]) -> go.Figure:
    """リソース使用状況チャートを作成"""
    if not metrics:
        return None
    
    timestamps = [m.get('timestamp', '') for m in metrics]
    gpu_usages = [m.get('gpu_usage', 0) * 100 for m in metrics]
    gpu_mem_usages = [m.get('gpu_memory_usage', 0) * 100 for m in metrics]
    cpu_usages = [m.get('cpu_usage', 0) * 100 for m in metrics]
    mem_usages = [m.get('memory_usage', 0) * 100 for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=gpu_usages,
        mode='lines',
        name='GPU Usage (%)',
        line=dict(color='#00ff41', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=gpu_mem_usages,
        mode='lines',
        name='GPU Memory (%)',
        line=dict(color='#00ffff', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cpu_usages,
        mode='lines',
        name='CPU Usage (%)',
        line=dict(color='#ff0040', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=mem_usages,
        mode='lines',
        name='Memory Usage (%)',
        line=dict(color='#ffaa00', width=2)
    ))
    
    fig.update_layout(
        title='Resource Usage Over Time',
        xaxis_title='Time',
        yaxis_title='Usage (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#00ff41', 'family': 'Orbitron'},
        legend=dict(bgcolor='rgba(0,0,0,0.8)', bordercolor='#00ff41'),
        height=400
    )
    
    return fig


def load_ppo_training_progress(log_file_path: Path) -> Dict:
    """PPOトレーニングの進捗をログファイルから読み込む"""
    try:
        if not log_file_path.exists():
            return {"status": "no_log_file", "message": f"Log file not found: {log_file_path}"}

        progress_data = {
            "status": "running",
            "current_step": 0,
            "max_steps": 100,
            "loss_history": [],
            "reward_history": [],
            "alpha_history": [],
            "kl_history": [],
            "last_update": None,
            "total_time": 0,
            "learning_rate": 0,
            "model_info": {}
        }

        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-200:]  # 最新の200行のみ読み込み

        for line in lines:
            # ステップ情報
            if 'step' in line.lower() and 'loss' in line.lower():
                try:
                    # INFO - Step 10/100: loss=2.145, reward=0.234, alpha=0.012, kl=0.045
                    parts = line.split('Step ')[-1].split('/')[0]
                    if parts.isdigit():
                        progress_data["current_step"] = int(parts)
                except:
                    pass

            # 最大ステップ数
            if 'max steps:' in line.lower():
                try:
                    max_steps_part = line.split('max steps:')[-1].strip()
                    if max_steps_part.isdigit():
                        progress_data["max_steps"] = int(max_steps_part)
                except:
                    pass

            # 損失値
            if 'loss=' in line:
                try:
                    loss_part = line.split('loss=')[-1].split(',')[0]
                    progress_data["loss_history"].append(float(loss_part))
                except:
                    pass

            # 報酬値
            if 'reward=' in line:
                try:
                    reward_part = line.split('reward=')[-1].split(',')[0]
                    progress_data["reward_history"].append(float(reward_part))
                except:
                    pass

            # Alpha値
            if 'alpha=' in line:
                try:
                    alpha_part = line.split('alpha=')[-1].split(',')[0]
                    progress_data["alpha_history"].append(float(alpha_part))
                except:
                    pass

            # KLダイバージェンス
            if 'kl=' in line:
                try:
                    kl_part = line.split('kl=')[-1].split(',')[0]
                    progress_data["kl_history"].append(float(kl_part))
                except:
                    pass

            # 学習率
            if 'lr=' in line:
                try:
                    lr_part = line.split('lr=')[-1].split(',')[0]
                    progress_data["learning_rate"] = float(lr_part)
                except:
                    pass

            # タイムスタンプ
            if line.startswith('2025-'):
                try:
                    timestamp_str = line.split(' - ')[0]
                    progress_data["last_update"] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                except:
                    pass

        # ステータス判定
        if progress_data["current_step"] >= progress_data["max_steps"]:
            progress_data["status"] = "completed"
        elif progress_data["current_step"] > 0:
            progress_data["status"] = "running"
        else:
            progress_data["status"] = "starting"

        # 履歴データの最新100個のみ保持
        for key in ["loss_history", "reward_history", "alpha_history", "kl_history"]:
            if len(progress_data[key]) > 100:
                progress_data[key] = progress_data[key][-100:]

        return progress_data

    except Exception as e:
        logger.error(f"Error loading PPO training progress: {e}")
        return {"status": "error", "message": str(e)}


def get_rtx3060_gpu_info() -> Dict:
    """RTX3060のGPU情報を取得"""
    try:
        import torch
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else -1,
            "memory_allocated": 0,
            "memory_reserved": 0,
            "memory_free": 0,
            "memory_total": 0,
            "utilization": 0
        }

        if torch.cuda.is_available():
            gpu_info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_info["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3    # GB
            gpu_info["memory_free"] = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - gpu_info["memory_reserved"]
            gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

            # 利用率取得（pynvmlが利用可能な場合）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info["utilization"] = util.gpu
                pynvml.nvmlShutdown()
            except ImportError:
                gpu_info["utilization"] = -1  # pynvml not available
            except Exception as e:
                logger.warning(f"Could not get GPU utilization: {e}")
                gpu_info["utilization"] = -1

        return gpu_info

    except Exception as e:
        logger.error(f"Error getting RTX3060 GPU info: {e}")
        return {"error": str(e)}


def main():
    """メイン関数"""
    # タイトル
    st.markdown('<h1 class="glitch-text">⚡ SO8T PIPELINE MONITOR</h1>', unsafe_allow_html=True)
    
    # 設定読み込み
    config_path = PROJECT_ROOT / "configs" / "master_automated_pipeline.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    checkpoint_dir = Path(config.get('pipeline', {}).get('checkpoint_dir', 'D:/webdataset/checkpoints/master_pipeline'))
    logs_dir = Path("_docs/progress_logs")
    screenshots_dir = Path("D:/webdataset/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("logs/resource_balancer")
    pipeline_log_file = Path(config.get('logging', {}).get('log_file', 'D:/webdataset/pipeline_logs/master_automated_pipeline.log'))
    
    # サイドバー
    with st.sidebar:
        st.markdown('<h2>⚙️ CONTROL PANEL</h2>', unsafe_allow_html=True)
        
        # 更新間隔設定
        refresh_interval = st.slider("更新間隔（秒）", 1, 30, 5)
        
        # 手動更新ボタン
        if st.button("🔄 FORCE REFRESH", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # 自動更新チェックボックス
        auto_refresh = st.checkbox("🔄 AUTO REFRESH", value=True)
        
        st.markdown("---")
        
        # セッション情報
        st.markdown('<h3>[STATS] SESSION INFO</h3>', unsafe_allow_html=True)
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            st.write(f"**Session ID**: `{checkpoint.get('session_id', 'N/A')}`")
            st.write(f"**Current Phase**: `{checkpoint.get('current_phase', 'N/A')}`")
            timestamp = checkpoint.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.write(f"**Last Update**: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    st.write(f"**Last Update**: {timestamp}")
    
    # メインコンテンツ
    if checkpoint is None:
        st.error("[WARN] No checkpoint found. Pipeline may not be running.")
        return
    
    # パイプライン状態セクション
    st.markdown('<h2>📡 PIPELINE STATUS</h2>', unsafe_allow_html=True)
    
    phase_results = checkpoint.get('phase_results', {})
    
    # フェーズ別ステータス表示
    cols = st.columns(5)
    phase_names = [
        ('phase0_dependencies', 'Phase 0'),
        ('phase1_web_scraping', 'Phase 1'),
        ('phase2_data_cleansing', 'Phase 2'),
        ('phase3_modeling_so8t', 'Phase 3'),
        ('phase4_integration', 'Phase 4')
    ]
    
    for i, (phase_key, phase_label) in enumerate(phase_names):
        with cols[i]:
            phase_data = phase_results.get(phase_key, {})
            status = phase_data.get('status', 'pending')
            color = get_phase_status_color(status)
            
            st.markdown(
                f'<div class="cyber-border">'
                f'<h3 style="color: {color};">{phase_label}</h3>'
                f'<p style="color: {color}; font-size: 1.2em;">{status.upper()}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    cols = st.columns(5)
    phase_names = [
        ('phase5_qlora_training', 'Phase 5'),
        ('phase6_evaluation', 'Phase 6'),
        ('phase7_ab_test', 'Phase 7'),
        ('phase8_post_processing', 'Phase 8'),
        ('phase9_japanese_test', 'Phase 9')
    ]
    
    for i, (phase_key, phase_label) in enumerate(phase_names):
        with cols[i]:
            phase_data = phase_results.get(phase_key, {})
            status = phase_data.get('status', 'pending')
            color = get_phase_status_color(status)
            
            st.markdown(
                f'<div class="cyber-border">'
                f'<h3 style="color: {color};">{phase_label}</h3>'
                f'<p style="color: {color}; font-size: 1.2em;">{status.upper()}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # タブでセクションを分ける
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "[STATS] Overview",
        "📈 Timeline",
        "💻 Resources",
        "[NG] Errors",
        "[NOTE] Logs",
        "🤖 PPO Training"
    ])
    
    with tab1:
        # 進捗統計
        st.markdown('<h2>[STATS] PROGRESS STATISTICS</h2>', unsafe_allow_html=True)
        
        completed_phases = sum(1 for p in phase_results.values() if p.get('status') == 'completed')
        running_phases = sum(1 for p in phase_results.values() if p.get('status') == 'running')
        failed_phases = sum(1 for p in phase_results.values() if p.get('status') == 'failed')
        total_phases = len(phase_results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            progress = (completed_phases / total_phases * 100) if total_phases > 0 else 0
            fig = create_cyberpunk_gauge(progress, 100, "Overall Progress", '#00ff41')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_cyberpunk_gauge(completed_phases, total_phases, "Completed", '#00ff41')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = create_cyberpunk_gauge(running_phases, total_phases, "Running", '#00ffff')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            fig = create_cyberpunk_gauge(failed_phases, total_phases, "Failed", '#ff0040')
            st.plotly_chart(fig, use_container_width=True)
        
        # 各フェーズの詳細進捗
        st.markdown('<h2>🔍 PHASE DETAILS</h2>', unsafe_allow_html=True)
        
        phase_labels = {
            'phase0_dependencies': 'Phase 0: Dependencies',
            'phase1_web_scraping': 'Phase 1: Web Scraping',
            'phase2_data_cleansing': 'Phase 2: Data Cleansing',
            'phase3_modeling_so8t': 'Phase 3: SO8T Modeling',
            'phase4_integration': 'Phase 4: Integration',
            'phase5_qlora_training': 'Phase 5: QLoRA Training',
            'phase6_evaluation': 'Phase 6: Evaluation',
            'phase7_ab_test': 'Phase 7: A/B Test',
            'phase8_post_processing': 'Phase 8: Post Processing',
            'phase9_japanese_test': 'Phase 9: Japanese Test'
        }
        
        for phase_key, phase_label in phase_labels.items():
            phase_data = phase_results.get(phase_key, {})
            status = phase_data.get('status', 'pending')
            color = get_phase_status_color(status)
            
            with st.expander(f"<span style='color: {color};'>{phase_label} - {status.upper()}</span>", expanded=(status == 'running')):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status**: <span style='color: {color};'>{status.upper()}</span>", unsafe_allow_html=True)
                    
                    result = phase_data.get('result', {})
                    if isinstance(result, dict):
                        output = result.get('output', result.get('status', 'N/A'))
                        st.write(f"**Output**: `{output}`")
                    elif result:
                        st.write(f"**Result**: `{result}`")
                    
                    attempt = phase_data.get('attempt', phase_data.get('attempts', 1))
                    st.write(f"**Attempt**: {attempt}")
                    
                    # 進捗情報
                    if isinstance(result, dict):
                        progress = result.get('progress', 0)
                        if progress > 0:
                            st.progress(progress)
                
                with col2:
                    error = phase_data.get('error')
                    if error:
                        st.error(f"**Error**: {str(error)[:200]}")
                    
                    if status == 'running':
                        st.info("Phase is currently running...")
                    elif status == 'completed':
                        st.success("Phase completed successfully!")
                    elif status == 'failed':
                        st.error("Phase failed. Check error details.")
    
    with tab2:
        # タイムライン表示
        st.markdown('<h2>⏱️ PIPELINE TIMELINE</h2>', unsafe_allow_html=True)
        
        fig = create_timeline_chart(phase_results)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # リソース使用状況
        st.markdown('<h2>💻 RESOURCE USAGE</h2>', unsafe_allow_html=True)
        
        metrics = load_resource_metrics(metrics_dir)
        
        if metrics:
            fig = create_resource_chart(metrics)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # 最新のメトリクスを表示
            if metrics:
                latest = metrics[-1]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("GPU Usage", f"{latest.get('gpu_usage', 0) * 100:.1f}%")
                
                with col2:
                    st.metric("GPU Memory", f"{latest.get('gpu_memory_usage', 0) * 100:.1f}%")
                
                with col3:
                    st.metric("CPU Usage", f"{latest.get('cpu_usage', 0) * 100:.1f}%")
                
                with col4:
                    st.metric("Memory Usage", f"{latest.get('memory_usage', 0) * 100:.1f}%")
        else:
            st.info("No resource metrics available.")
    
    with tab4:
        # エラー詳細表示
        st.markdown('<h2>[NG] ERROR DETAILS</h2>', unsafe_allow_html=True)
        
        failed_phases = {k: v for k, v in phase_results.items() if v.get('status') == 'failed'}
        
        if failed_phases:
            for phase_key, phase_data in failed_phases.items():
                phase_label = phase_labels.get(phase_key, phase_key)
                error = phase_data.get('error', 'No error message')
                attempts = phase_data.get('attempts', phase_data.get('attempt', 1))
                
                st.error(f"**{phase_label}**")
                st.code(str(error), language='text')
                st.write(f"**Attempts**: {attempts}")
                
                # エラーの詳細情報
                result = phase_data.get('result', {})
                if result:
                    st.write(f"**Result**: {result}")
                
                st.markdown("---")
        else:
            st.success("No errors detected. All phases completed successfully!")
    
    with tab5:
        # ログのリアルタイムストリーミング
        st.markdown('<h2>[NOTE] PIPELINE LOGS</h2>', unsafe_allow_html=True)
        
        log_lines = load_pipeline_log(pipeline_log_file, max_lines=200)
        
        if log_lines:
            # ログを表示（最新のものから）
            log_text = ''.join(log_lines)
            st.code(log_text, language='text')
            
            # ログファイルの更新時刻
            if pipeline_log_file.exists():
                mtime = datetime.fromtimestamp(pipeline_log_file.stat().st_mtime)
                st.caption(f"Last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No log file available.")
        
        # 進捗ログセクション
        st.markdown('<h3>📋 PROGRESS LOGS</h3>', unsafe_allow_html=True)
        
        logs = load_progress_logs(logs_dir)
        
        if logs:
            # 最新のログを表示
            latest_log = logs[0]
            st.markdown(f'<p style="color: #00ff41;">Latest Log: {latest_log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
            
            # ログ内容を表示（最初の2000文字）
            log_content = latest_log['content'][:2000]
            st.code(log_content, language='markdown')
            
            # ログファイル一覧
            with st.expander("📋 Log Files"):
                for log in logs[:10]:
                    st.write(f"- **{log['file']}** ({log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            st.info("No progress logs available.")
    
    # ブラウザスクリーンショットセクション（メインコンテンツに表示）
    st.markdown('<h2>🌐 BROWSER VIEW</h2>', unsafe_allow_html=True)
    
    screenshots = load_browser_screenshots(screenshots_dir)
    
    if screenshots:
        # 最新のスクリーンショットを大きく表示
        latest_screenshot = screenshots[0]
        st.markdown(f'<p style="color: #00ff41;">Latest Screenshot: {latest_screenshot["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
        st.image(latest_screenshot['image'], use_container_width=True, caption=f"Browser View - {latest_screenshot['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 過去のスクリーンショットをサムネイル表示
        if len(screenshots) > 1:
            st.markdown('<h3>📸 Screenshot History</h3>', unsafe_allow_html=True)
            cols = st.columns(min(len(screenshots) - 1, 4))
            for i, screenshot in enumerate(screenshots[1:5]):
                with cols[i % 4]:
                    st.image(screenshot['image'], use_container_width=True, caption=screenshot['timestamp'].strftime('%H:%M:%S'))
    else:
        st.info("No browser screenshots available. Browser capture may not be active.")
    
    # 自動更新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

    with tab6:
        # PPOトレーニング進捗
        st.markdown('<h2>🤖 PPO TRAINING PROGRESS</h2>', unsafe_allow_html=True)

        # PPOログファイルのパス
        ppo_log_file = PROJECT_ROOT / "logs" / "aegis_v2_ppo_training.log"

        # RTX3060 GPU情報取得
        gpu_info = get_rtx3060_gpu_info()

        # PPOトレーニング進捗読み込み
        ppo_progress = load_ppo_training_progress(ppo_log_file)

        if ppo_progress["status"] == "no_log_file":
            st.warning("PPOトレーニングログファイルが見つかりません")
            st.info(f"期待される場所: {ppo_log_file}")
            return

        elif ppo_progress["status"] == "error":
            st.error(f"PPOトレーニング進捗の読み込みに失敗しました: {ppo_progress.get('message', 'Unknown error')}")
            return

        # ステータス表示
        status_color = {
            "starting": "#ffff00",  # 黄色
            "running": "#00ff41",   # 緑
            "completed": "#0080ff", # 青
            "error": "#ff0040"      # 赤
        }.get(ppo_progress["status"], "#ffffff")

        st.markdown(
            f'<div class="cyber-border" style="text-align: center;">'
            f'<h3 style="color: {status_color}; font-size: 2em;">{ppo_progress["status"].upper()}</h3>'
            f'<p style="color: #00ff41; font-size: 1.5em;">Step {ppo_progress["current_step"]}/{ppo_progress["max_steps"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # 進捗バー
        if ppo_progress["max_steps"] > 0:
            progress = min(ppo_progress["current_step"] / ppo_progress["max_steps"], 1.0)
            st.progress(progress)

        # RTX3060 GPU情報表示
        if isinstance(gpu_info, dict) and not gpu_info.get("error"):
            st.markdown('<h3>🎮 RTX3060 GPU Status</h3>', unsafe_allow_html=True)

            gpu_col1, gpu_col2, gpu_col3, gpu_col4 = st.columns(4)

            with gpu_col1:
                mem_used = gpu_info.get("memory_allocated", 0)
                mem_total = gpu_info.get("memory_total", 0)
                st.metric("VRAM Used", ".1f", ".1f")

            with gpu_col2:
                mem_free = gpu_info.get("memory_free", 0)
                st.metric("VRAM Free", ".1f")

            with gpu_col3:
                utilization = gpu_info.get("utilization", -1)
                if utilization >= 0:
                    st.metric("GPU Util", f"{utilization}%")
                else:
                    st.metric("GPU Util", "N/A")

            with gpu_col4:
                device_count = gpu_info.get("device_count", 0)
                st.metric("GPU Count", device_count)

            # VRAM使用率バー
            if mem_total > 0:
                vram_usage = (mem_used / mem_total) * 100
                st.progress(min(vram_usage / 100, 1.0))
                st.caption(".1f")

        # トレーニングメトリクス表示
        st.markdown('<h3>[STATS] Training Metrics</h3>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            current_loss = ppo_progress["loss_history"][-1] if ppo_progress["loss_history"] else 0.0
            st.metric("Current Loss", ".4f")

        with col2:
            current_reward = ppo_progress["reward_history"][-1] if ppo_progress["reward_history"] else 0.0
            st.metric("Current Reward", ".4f")

        with col3:
            current_alpha = ppo_progress["alpha_history"][-1] if ppo_progress["alpha_history"] else 0.0
            st.metric("Alpha", ".4f")

        with col4:
            lr = ppo_progress["learning_rate"]
            st.metric("Learning Rate", ".2e")

        # グラフ表示
        st.markdown('<h3>📈 Training Metrics</h3>', unsafe_allow_html=True)

        # 損失と報酬のグラフ
        if ppo_progress["loss_history"] or ppo_progress["reward_history"]:
            fig = go.Figure()

            if ppo_progress["loss_history"]:
                steps = list(range(len(ppo_progress["loss_history"])))
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=ppo_progress["loss_history"],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='#ff0040', width=2)
                ))

            if ppo_progress["reward_history"]:
                steps = list(range(len(ppo_progress["reward_history"])))
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=ppo_progress["reward_history"],
                    mode='lines+markers',
                    name='Reward',
                    line=dict(color='#00ff41', width=2),
                    yaxis='y2'
                ))

            fig.update_layout(
                title="Loss & Reward History",
                xaxis_title="Steps",
                yaxis_title="Loss",
                yaxis2=dict(title="Reward", overlaying="y", side="right"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.1)",
                font=dict(color="#00ff41")
            )

            st.plotly_chart(fig, use_container_width=True)

        # AlphaとKLダイバージェンスのグラフ
        if ppo_progress["alpha_history"] or ppo_progress["kl_history"]:
            fig2 = go.Figure()

            if ppo_progress["alpha_history"]:
                steps = list(range(len(ppo_progress["alpha_history"])))
                fig2.add_trace(go.Scatter(
                    x=steps,
                    y=ppo_progress["alpha_history"],
                    mode='lines+markers',
                    name='Alpha',
                    line=dict(color='#0080ff', width=2)
                ))

            if ppo_progress["kl_history"]:
                steps = list(range(len(ppo_progress["kl_history"])))
                fig2.add_trace(go.Scatter(
                    x=steps,
                    y=ppo_progress["kl_history"],
                    mode='lines+markers',
                    name='KL Divergence',
                    line=dict(color='#ffff00', width=2),
                    yaxis='y2'
                ))

            fig2.update_layout(
                title="Alpha & KL Divergence History",
                xaxis_title="Steps",
                yaxis_title="Alpha",
                yaxis2=dict(title="KL", overlaying="y", side="right"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.1)",
                font=dict(color="#00ff41")
            )

            st.plotly_chart(fig2, use_container_width=True)

        # 最新ログ表示
        st.markdown('<h3>[NOTE] Recent Logs</h3>', unsafe_allow_html=True)

        try:
            if ppo_log_file.exists():
                with open(ppo_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[-20:]  # 最新20行

                log_text = ''.join(lines)
                st.code(log_text, language='log')
            else:
                st.warning("ログファイルが見つかりません")
        except Exception as e:
            st.error(f"ログ読み込みエラー: {e}")


if __name__ == "__main__":
    main()
