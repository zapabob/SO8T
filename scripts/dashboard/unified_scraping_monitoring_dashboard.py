#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合スクレイピング監視ダッシュボード

全ブラウザインスタンスの状態監視、リアルタイムスクリーンショット、SO8T統制判断結果表示

Usage:
    streamlit run scripts/dashboard/unified_scraping_monitoring_dashboard.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from PIL import Image

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "audit"))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("[ERROR] Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

# 監査ログインポート
try:
    from scripts.audit.scraping_audit_logger import ScrapingAuditLogger
    AUDIT_LOGGER_AVAILABLE = True
except ImportError:
    AUDIT_LOGGER_AVAILABLE = False

# ロギング設定
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedScrapingMonitoringDashboard:
    """統合スクレイピング監視ダッシュボード"""
    
    def __init__(self):
        """初期化"""
        self.output_dir = Path("D:/webdataset/processed")
        self.log_dir = Path("logs")
        self.checkpoint_dir = Path("D:/webdataset/checkpoints/power_failure_recovery")
        self.screenshots_dir = self.output_dir / "screenshots"
        
        # 監査ロガー初期化
        self.audit_logger = None
        if AUDIT_LOGGER_AVAILABLE:
            try:
                self.audit_logger = ScrapingAuditLogger()
            except Exception as e:
                logger.warning(f"Failed to initialize audit logger: {e}")
        
        # セッション状態の初期化
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
    
    def load_browser_status(self) -> Dict[int, Dict]:
        """ブラウザ状態を読み込み"""
        browser_status = {}
        
        # ダッシュボード状態ファイルから読み込み
        state_file = self.output_dir / "dashboard_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    browser_status = state.get('browser_status', {})
            except Exception as e:
                logger.error(f"Failed to load browser status: {e}")
        
        # 監査ログからも読み込み
        if self.audit_logger:
            try:
                active_sessions = self.audit_logger.get_active_sessions()
                for session in active_sessions:
                    browser_index = session.get('browser_index', 0)
                    browser_status[browser_index] = {
                        'status': session.get('status', 'active'),
                        'current_keyword': session.get('keyword', ''),
                        'samples_collected': session.get('samples_collected', 0),
                        'last_activity': session.get('last_activity', ''),
                        'session_id': session.get('session_id', '')
                    }
            except Exception as e:
                logger.error(f"Failed to load browser status from audit log: {e}")
        
        return browser_status
    
    def load_so8t_decisions(self) -> List[Dict]:
        """SO8T統制判断結果を読み込み"""
        decisions = []
        
        # ダッシュボード状態ファイルから読み込み
        state_file = self.output_dir / "dashboard_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    decisions = state.get('so8t_decisions', [])
            except Exception as e:
                logger.error(f"Failed to load SO8T decisions: {e}")
        
        # 監査ログからも読み込み
        if self.audit_logger:
            try:
                stats = self.audit_logger.get_statistics()
                # 最新のSO8T判断イベントを取得
                # TODO: 監査ログからSO8T判断イベントを取得する機能を追加
            except Exception as e:
                logger.error(f"Failed to load SO8T decisions from audit log: {e}")
        
        return decisions
    
    def load_latest_samples(self) -> List[Dict]:
        """最新のサンプルを読み込み"""
        samples = []
        if self.output_dir.exists():
            # 最新のJSONLファイルを探す
            jsonl_files = sorted(
                self.output_dir.glob("*.jsonl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if jsonl_files:
                latest_file = jsonl_files[0]
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                samples.append(json.loads(line))
                except Exception as e:
                    logger.error(f"Failed to load samples: {e}")
        
        return samples
    
    def load_screenshots(self) -> Dict[int, str]:
        """スクリーンショットを読み込み"""
        screenshots = {}
        
        if self.screenshots_dir.exists():
            # ブラウザごとの最新スクリーンショットを取得
            for browser_index in range(10):  # 最大10ブラウザ
                browser_screenshots = sorted(
                    self.screenshots_dir.glob(f"browser_{browser_index}_*.png"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                if browser_screenshots:
                    screenshots[browser_index] = str(browser_screenshots[0])
        
        return screenshots
    
    def render_dashboard(self):
        """ダッシュボードをレンダリング"""
        st.set_page_config(
            page_title="SO8T統制Webスクレイピング統合監視ダッシュボード",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 SO8T統制Webスクレイピング統合監視ダッシュボード")
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
        
        # データ読み込み
        with st.spinner("データを読み込み中..."):
            browser_status = self.load_browser_status()
            so8t_decisions = self.load_so8t_decisions()
            latest_samples = self.load_latest_samples()
            screenshots = self.load_screenshots()
        
        # 全体統計
        st.subheader("📊 全体統計")
        col1, col2, col3, col4 = st.columns(4)
        
        total_samples = len(latest_samples)
        nsfw_samples = sum(1 for s in latest_samples if s.get('nsfw_label') != 'safe')
        active_browsers = sum(1 for s in browser_status.values() if s.get('status') == 'active')
        completed_browsers = sum(1 for s in browser_status.values() if s.get('status') == 'completed')
        
        with col1:
            st.metric("総サンプル数", f"{total_samples:,}")
        with col2:
            st.metric("NSFW検知サンプル", f"{nsfw_samples:,}")
        with col3:
            st.metric("アクティブブラウザ", f"{active_browsers}")
        with col4:
            st.metric("完了ブラウザ", f"{completed_browsers}")
        
        st.markdown("---")
        
        # ブラウザ状態表示
        st.subheader("🌐 ブラウザ状態")
        
        if browser_status:
            # ブラウザ状態テーブル
            browser_df_data = []
            for browser_num, status in browser_status.items():
                browser_df_data.append({
                    'ブラウザ番号': browser_num,
                    '状態': status.get('status', 'unknown'),
                    '処理中キーワード': status.get('current_keyword', 'なし'),
                    '収集サンプル数': status.get('samples_collected', 0),
                    '最終活動': status.get('last_activity', 'なし')[:19] if status.get('last_activity') else 'なし',
                    'セッションID': status.get('session_id', '')[:16] + '...' if status.get('session_id') else ''
                })
            
            browser_df = pd.DataFrame(browser_df_data)
            st.dataframe(browser_df, use_container_width=True)
            
            # ブラウザ状態の可視化
            col1, col2 = st.columns(2)
            
            with col1:
                # 状態別ブラウザ数
                status_counts = {}
                for status in browser_status.values():
                    s = status.get('status', 'unknown')
                    status_counts[s] = status_counts.get(s, 0) + 1
                
                if status_counts:
                    st.bar_chart(status_counts)
            
            with col2:
                # サンプル収集数の可視化
                samples_by_browser = {
                    f"Browser {num}": status.get('samples_collected', 0)
                    for num, status in browser_status.items()
                }
                if samples_by_browser:
                    st.bar_chart(samples_by_browser)
        else:
            st.info("ブラウザ状態データが見つかりません")
        
        st.markdown("---")
        
        # ブラウザスクリーンショット表示
        st.subheader("📸 ブラウザスクリーンショット（リアルタイム）")
        
        if screenshots:
            # ブラウザごとにスクリーンショットを表示
            cols = st.columns(min(len(screenshots), 5))
            for idx, (browser_index, screenshot_path) in enumerate(list(screenshots.items())[:5]):
                with cols[idx % 5]:
                    try:
                        img = Image.open(screenshot_path)
                        st.image(img, caption=f"Browser {browser_index}", use_container_width=True)
                        st.caption(f"最終更新: {Path(screenshot_path).stat().st_mtime}")
                    except Exception as e:
                        st.error(f"Browser {browser_index}: 画像読み込み失敗")
        else:
            st.info("スクリーンショットが見つかりません")
        
        st.markdown("---")
        
        # SO8T統制判断結果表示
        st.subheader("🤖 SO8T統制判断結果")
        
        if so8t_decisions:
            # 最新10件を表示
            recent_decisions = so8t_decisions[-10:]
            
            for decision in reversed(recent_decisions):
                decision_type = decision.get('type', 'unknown')
                decision_result = decision.get('decision', 'unknown')
                reasoning = decision.get('reasoning', '')
                keyword = decision.get('keyword', '')
                timestamp = decision.get('timestamp', '')
                
                # 判断結果に応じた色
                if decision_result == 'allow':
                    st.success(f"✅ [{decision_type}] {keyword} - {decision_result}")
                elif decision_result == 'deny':
                    st.error(f"❌ [{decision_type}] {keyword} - {decision_result}")
                elif decision_result == 'modify':
                    st.warning(f"⚠️ [{decision_type}] {keyword} - {decision_result}")
                else:
                    st.info(f"ℹ️ [{decision_type}] {keyword} - {decision_result}")
                
                if reasoning:
                    with st.expander("推論内容"):
                        st.text(reasoning[:500])
                
                st.caption(f"時刻: {timestamp[:19] if timestamp else '不明'}")
                st.markdown("---")
            
            # 判断タイプ別統計
            col1, col2 = st.columns(2)
            
            with col1:
                type_counts = {}
                for d in so8t_decisions:
                    t = d.get('type', 'unknown')
                    type_counts[t] = type_counts.get(t, 0) + 1
                if type_counts:
                    st.bar_chart(type_counts)
            
            with col2:
                decision_counts = {}
                for d in so8t_decisions:
                    dec = d.get('decision', 'unknown')
                    decision_counts[dec] = decision_counts.get(dec, 0) + 1
                if decision_counts:
                    st.bar_chart(decision_counts)
        else:
            st.info("SO8T統制判断結果が見つかりません")
        
        st.markdown("---")
        
        # 最新サンプル表示
        st.subheader("📝 最新サンプル")
        
        if latest_samples:
            # 最新10件を表示
            recent_samples = latest_samples[-10:]
            
            for sample in reversed(recent_samples):
                url = sample.get('url', '')
                keyword = sample.get('keyword', '')
                category = sample.get('category', '')
                language = sample.get('language', '')
                text_preview = sample.get('text', '')[:200]
                nsfw_label = sample.get('nsfw_label', 'safe')
                
                st.markdown(f"**URL**: {url[:80]}...")
                st.markdown(f"**キーワード**: {keyword} | **カテゴリ**: {category} | **言語**: {language}")
                st.markdown(f"**NSFWラベル**: {nsfw_label}")
                st.text(f"テキストプレビュー: {text_preview}...")
                st.markdown("---")
        else:
            st.info("サンプルが見つかりません")
        
        # 自動更新
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()


def main():
    """メイン関数"""
    dashboard = UnifiedScrapingMonitoringDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()

