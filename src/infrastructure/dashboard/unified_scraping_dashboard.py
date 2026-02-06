#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統制Webスクレイピング統一管理ダッシュボード

すべてのWebスクレイピングスクリプトを統合管理するStreamlitダッシュボード

Usage:
    streamlit run scripts/dashboard/unified_scraping_dashboard.py
"""

import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from PIL import Image

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

# ロギング設定
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedScrapingDashboard:
    """SO8T統制Webスクレイピング統一管理ダッシュボード"""
    
    def __init__(self):
        """初期化"""
        self.output_dir = Path("D:/webdataset/processed")
        self.log_dir = Path("logs")
        self.checkpoint_dir = Path("D:/webdataset/checkpoints/pipeline")
        
        # スクレイピングスクリプト定義
        self.scraping_scripts = {
            'parallel_deep_research': {
                'name': '並列DeepResearch Webスクレイピング',
                'script': 'scripts/data/parallel_deep_research_scraping.py',
                'batch': 'scripts/data/run_parallel_deep_research_scraping.bat',
                'description': '10個のブラウザで並列実行、SO8T統制',
                'enabled': True
            },
            'arxiv_open_access': {
                'name': 'Arxiv・オープンアクセス論文スクレイピング',
                'script': 'scripts/data/arxiv_open_access_scraping.py',
                'batch': 'scripts/data/run_arxiv_background_scraping.bat',
                'description': 'Arxiv全ジャンルとオープンアクセス論文',
                'enabled': True
            },
            'auto_background': {
                'name': 'SO8T統制完全自動バックグラウンドスクレイピング',
                'script': 'scripts/data/so8t_auto_background_scraping.py',
                'batch': 'scripts/data/run_so8t_auto_background_scraping.bat',
                'description': '完全自動バックグラウンド実行',
                'enabled': True
            },
            'comprehensive_category': {
                'name': '包括的カテゴリWebスクレイピング',
                'script': 'scripts/data/comprehensive_category_scraping.py',
                'batch': 'scripts/data/run_comprehensive_category_scraping.bat',
                'description': '広範なカテゴリのスクレイピング',
                'enabled': True
            },
            'deep_research_category': {
                'name': 'DeepResearchカテゴリ別スクレイピング',
                'script': 'scripts/data/deep_research_category_scraping.py',
                'batch': 'scripts/data/run_deep_research_scraping.bat',
                'description': 'DeepResearchによるキーワード検索',
                'enabled': True
            }
        }
        
        # セッション状態の初期化
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'running_processes' not in st.session_state:
            st.session_state.running_processes = {}
        if 'error_logs' not in st.session_state:
            st.session_state.error_logs = []
    
    def check_process_status(self, script_key: str) -> Dict:
        """プロセス状態をチェック"""
        status = {
            'running': False,
            'pid': None,
            'start_time': None,
            'error_count': 0,
            'last_error': None
        }
        
        # ログファイルから状態を確認
        log_file = self.log_dir / f"{script_key}.log"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        # 最後の行から状態を確認
                        last_line = lines[-1]
                        if 'ERROR' in last_line or 'FAILED' in last_line:
                            status['error_count'] += 1
                            status['last_error'] = last_line.strip()
                        elif 'SUCCESS' in last_line or 'completed' in last_line.lower():
                            status['running'] = False
                        else:
                            # 最近のログがあれば実行中と判断
                            if len(lines) > 10:
                                status['running'] = True
            except Exception as e:
                logger.error(f"Failed to read log file: {e}")
        
        return status
    
    def load_all_status(self) -> Dict[str, Dict]:
        """すべてのスクレイピングスクリプトの状態を読み込み"""
        all_status = {}
        
        for script_key, script_info in self.scraping_scripts.items():
            if script_info['enabled']:
                status = self.check_process_status(script_key)
                all_status[script_key] = {
                    **status,
                    'name': script_info['name'],
                    'description': script_info['description']
                }
        
        return all_status
    
    def load_error_logs(self) -> List[Dict]:
        """エラーログを読み込み"""
        error_logs = []
        
        # 各スクリプトのログからエラーを抽出
        for script_key in self.scraping_scripts.keys():
            log_file = self.log_dir / f"{script_key}.log"
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if 'ERROR' in line or '404' in line or '200' in line:
                                error_logs.append({
                                    'timestamp': datetime.now().isoformat(),
                                    'script': script_key,
                                    'error': line.strip(),
                                    'line_number': i + 1
                                })
                except Exception as e:
                    logger.error(f"Failed to read error log: {e}")
        
        # 時系列でソート
        error_logs.sort(key=lambda x: x['timestamp'], reverse=True)
        return error_logs[:100]  # 最新100件
    
    def start_scraping_script(self, script_key: str) -> bool:
        """スクレイピングスクリプトを開始"""
        if script_key not in self.scraping_scripts:
            return False
        
        script_info = self.scraping_scripts[script_key]
        batch_file = PROJECT_ROOT / script_info['batch']
        
        if not batch_file.exists():
            logger.error(f"Batch file not found: {batch_file}")
            return False
        
        try:
            # バックグラウンドで実行
            process = subprocess.Popen(
                [str(batch_file)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            st.session_state.running_processes[script_key] = {
                'pid': process.pid,
                'start_time': datetime.now().isoformat(),
                'process': process
            }
            
            logger.info(f"Started scraping script: {script_key} (PID: {process.pid})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start script {script_key}: {e}")
            return False
    
    def stop_scraping_script(self, script_key: str) -> bool:
        """スクレイピングスクリプトを停止"""
        if script_key not in st.session_state.running_processes:
            return False
        
        try:
            process_info = st.session_state.running_processes[script_key]
            process = process_info.get('process')
            
            if process:
                process.terminate()
                process.wait(timeout=10)
            
            del st.session_state.running_processes[script_key]
            logger.info(f"Stopped scraping script: {script_key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to stop script {script_key}: {e}")
            return False
    
    def render_dashboard(self):
        """ダッシュボードをレンダリング"""
        st.set_page_config(
            page_title="SO8T統制Webスクレイピング統一管理ダッシュボード",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 SO8T統制Webスクレイピング統一管理ダッシュボード")
        st.markdown("---")
        
        # 自動更新設定
        col1, col2, col3 = st.columns(3)
        with col1:
            auto_refresh = st.checkbox("自動更新", value=True)
        with col2:
            refresh_interval = st.slider("更新間隔（秒）", 1, 60, 5)
        with col3:
            if st.button("🔄 手動更新"):
                st.session_state.last_update = datetime.now()
                st.rerun()
        
        # 自動更新
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        
        # データ読み込み
        with st.spinner("データを読み込み中..."):
            all_status = self.load_all_status()
            error_logs = self.load_error_logs()
        
        # 全体統計
        st.subheader("[STATS] 全体統計")
        col1, col2, col3, col4 = st.columns(4)
        
        total_scripts = len(all_status)
        running_scripts = sum(1 for s in all_status.values() if s['running'])
        error_scripts = sum(1 for s in all_status.values() if s['error_count'] > 0)
        total_errors = sum(s['error_count'] for s in all_status.values())
        
        with col1:
            st.metric("総スクリプト数", total_scripts)
        with col2:
            st.metric("実行中", running_scripts, delta=f"{running_scripts/total_scripts*100:.1f}%" if total_scripts > 0 else "0%")
        with col3:
            st.metric("エラー発生", error_scripts, delta=f"{error_scripts/total_scripts*100:.1f}%" if total_scripts > 0 else "0%")
        with col4:
            st.metric("総エラー数", total_errors)
        
        st.markdown("---")
        
        # スクレイピングスクリプト管理
        st.subheader("🌐 スクレイピングスクリプト管理")
        
        for script_key, status in all_status.items():
            script_info = self.scraping_scripts[script_key]
            
            with st.expander(f"{status['name']} - {'🟢 実行中' if status['running'] else '🔴 停止中'}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**説明**: {status['description']}")
                    if status['running']:
                        st.write(f"**状態**: 🟢 実行中")
                    else:
                        st.write(f"**状態**: 🔴 停止中")
                    
                    if status['error_count'] > 0:
                        st.warning(f"**エラー数**: {status['error_count']}")
                        if status['last_error']:
                            st.error(f"**最後のエラー**: {status['last_error']}")
                
                with col2:
                    if script_key in st.session_state.running_processes:
                        if st.button("停止", key=f"stop_{script_key}"):
                            self.stop_scraping_script(script_key)
                            st.rerun()
                    else:
                        if st.button("開始", key=f"start_{script_key}"):
                            self.start_scraping_script(script_key)
                            st.rerun()
        
        st.markdown("---")
        
        # エラーログ表示
        st.subheader("🚨 エラーログ")
        
        if error_logs:
            error_df = pd.DataFrame(error_logs)
            st.dataframe(error_df, use_container_width=True)
            
            # エラー統計
            col1, col2 = st.columns(2)
            
            with col1:
                error_by_script = error_df.groupby('script').size()
                st.bar_chart(error_by_script)
            
            with col2:
                # エラータイプ別統計
                error_types = {}
                for error in error_logs:
                    error_text = error['error'].lower()
                    if '404' in error_text:
                        error_types['404 Not Found'] = error_types.get('404 Not Found', 0) + 1
                    elif '200' in error_text and 'empty' in error_text:
                        error_types['200 Empty Content'] = error_types.get('200 Empty Content', 0) + 1
                    elif 'timeout' in error_text:
                        error_types['Timeout'] = error_types.get('Timeout', 0) + 1
                    else:
                        error_types['Other'] = error_types.get('Other', 0) + 1
                
                if error_types:
                    st.bar_chart(error_types)
        else:
            st.info("エラーログがありません")
        
        st.markdown("---")
        
        # 全自動パイプライン制御
        st.subheader("⚙️ 全自動パイプライン制御")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("[START] 全スクリプト開始", use_container_width=True):
                for script_key in self.scraping_scripts.keys():
                    if script_key not in st.session_state.running_processes:
                        self.start_scraping_script(script_key)
                st.success("すべてのスクリプトを開始しました")
                st.rerun()
        
        with col2:
            if st.button("🛑 全スクリプト停止", use_container_width=True):
                for script_key in list(st.session_state.running_processes.keys()):
                    self.stop_scraping_script(script_key)
                st.success("すべてのスクリプトを停止しました")
                st.rerun()
        
        # フッター
        st.markdown("---")
        st.markdown(f"**最終更新**: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """メイン関数"""
    dashboard = UnifiedScrapingDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()





