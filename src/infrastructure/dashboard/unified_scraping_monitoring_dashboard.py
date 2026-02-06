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
    from src.audit.scraping_audit_logger import ScrapingAuditLogger
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
        
        # キーワード入力セクション
        st.subheader("🔍 キーワード検索")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            keyword_input = st.text_input(
                "キーワードを入力（カンマ区切りで複数入力可能）",
                placeholder="例: Python, Rust, TypeScript, JavaScript",
                help="複数のキーワードをカンマ区切りで入力できます"
            )
        
        # 優先度選択
        priority = st.selectbox(
            "優先度",
            ["low", "medium", "high", "urgent"],
            index=1,  # デフォルト: medium
            help="キーワードの優先度を選択してください"
        )
        
        with col2:
            st.write("")  # スペーサー
            st.write("")  # スペーサー
            if st.button("📤 キーワード送信", type="primary"):
                if keyword_input:
                    keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
                    if keywords:
                        try:
                            from src.utils.keyword_coordinator import KeywordCoordinator
                            coordinator = KeywordCoordinator()
                            added_count = coordinator.add_keywords(keywords, source="streamlit", priority=priority)
                            st.success(f"[OK] {added_count}個のキーワードを追加しました（優先度: {priority}）: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
                            st.session_state.last_update = datetime.now()
                        except Exception as e:
                            st.error(f"[NG] キーワード追加に失敗しました: {e}")
                    else:
                        st.warning("[WARN] 有効なキーワードが入力されていません")
                else:
                    st.warning("[WARN] キーワードを入力してください")
        
        # キーワード状態表示
        try:
            from src.utils.keyword_coordinator import KeywordCoordinator
            coordinator = KeywordCoordinator()
            stats = coordinator.get_statistics()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("総キーワード数", stats.get('total', 0))
            with col2:
                st.metric("待機中", stats.get('pending', 0))
            with col3:
                st.metric("処理中", stats.get('processing', 0))
            with col4:
                st.metric("完了", stats.get('completed', 0))
            with col5:
                st.metric("失敗", stats.get('failed', 0))
            
            # 優先度別統計
            priority_stats = stats.get('by_priority', {})
            if priority_stats:
                st.markdown("**優先度別キーワード数**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("低", priority_stats.get('low', 0))
                with col2:
                    st.metric("中", priority_stats.get('medium', 0))
                with col3:
                    st.metric("高", priority_stats.get('high', 0))
                with col4:
                    st.metric("緊急", priority_stats.get('urgent', 0))
            
            # 優先度フィルタ
            priority_filter = st.selectbox(
                "優先度でフィルタ",
                ["すべて", "low", "medium", "high", "urgent"],
                index=0
            )
            
            # 処理中のキーワード一覧
            filter_priority = None if priority_filter == "すべて" else priority_filter
            processing_keywords = coordinator.get_all_keywords(status_filter=None, priority_filter=filter_priority)
            if processing_keywords:
                st.markdown("**キーワード一覧**")
                keyword_df_data = []
                for kw_data in processing_keywords[-20:]:  # 最新20件
                    keyword_df_data.append({
                        'キーワード': kw_data.get('keyword', ''),
                        '優先度': kw_data.get('priority', 'medium'),
                        '状態': kw_data.get('status', 'unknown'),
                        'ブラウザID': kw_data.get('browser_id', 'なし'),
                        '追加時刻': kw_data.get('added_at', '')[:19] if kw_data.get('added_at') else '',
                        '割り当て時刻': kw_data.get('assigned_at', '')[:19] if kw_data.get('assigned_at') else '',
                    })
                
                if keyword_df_data:
                    keyword_df = pd.DataFrame(keyword_df_data)
                    st.dataframe(keyword_df, use_container_width=True)
            
            # 詳細統計セクション
            st.markdown("---")
            st.subheader("[STATS] キーワード詳細統計")
            
            progress_stats = stats.get('progress_stats', {})
            if progress_stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("総サンプル数", f"{progress_stats.get('total_samples', 0):,}")
                with col2:
                    st.metric("総URL処理数", f"{progress_stats.get('total_urls_processed', 0):,}")
                with col3:
                    st.metric("平均処理時間", f"{progress_stats.get('avg_processing_time', 0.0):.2f}秒")
                with col4:
                    st.metric("成功率", f"{progress_stats.get('success_rate', 0.0)*100:.1f}%")
            
            # 時間別の処理状況（時系列グラフ）
            by_time = stats.get('by_time', {})
            if by_time:
                st.markdown("**時間別の処理状況**")
                time_df = pd.DataFrame(list(by_time.items()), columns=['時刻', '処理数'])
                time_df = time_df.sort_values('時刻')
                st.line_chart(time_df.set_index('時刻'))
            
            # ブラウザ別の処理状況（棒グラフ）
            by_browser = stats.get('by_browser', {})
            if by_browser:
                st.markdown("**ブラウザ別の処理状況**")
                browser_df = pd.DataFrame(list(by_browser.items()), columns=['ブラウザID', 'キーワード数'])
                browser_df = browser_df.sort_values('ブラウザID')
                st.bar_chart(browser_df.set_index('ブラウザID'))
            
            # 優先度別の処理状況（積み上げ棒グラフ）
            by_priority = stats.get('by_priority', {})
            if by_priority:
                st.markdown("**優先度別の処理状況**")
                priority_df = pd.DataFrame(list(by_priority.items()), columns=['優先度', 'キーワード数'])
                st.bar_chart(priority_df.set_index('優先度'))
            
            # 統計情報のエクスポート
            st.markdown("---")
            st.markdown("**統計情報のエクスポート**")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📥 CSV形式でエクスポート"):
                    try:
                        # すべてのキーワードデータを取得
                        all_keywords = coordinator.get_all_keywords()
                        export_data = []
                        for kw_data in all_keywords:
                            progress = kw_data.get('progress', {})
                            export_data.append({
                                'キーワード': kw_data.get('keyword', ''),
                                '優先度': kw_data.get('priority', 'medium'),
                                '状態': kw_data.get('status', 'unknown'),
                                'ブラウザID': kw_data.get('browser_id', ''),
                                'サンプル数': progress.get('samples_collected', 0),
                                'URL処理数': progress.get('urls_processed', 0),
                                'URL失敗数': progress.get('urls_failed', 0),
                                '成功率': progress.get('success_rate', 0.0),
                                '追加時刻': kw_data.get('added_at', ''),
                                '完了時刻': kw_data.get('completed_at', '')
                            })
                        
                        export_df = pd.DataFrame(export_data)
                        csv = export_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 CSVダウンロード",
                            data=csv,
                            file_name=f"keyword_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"[NG] エクスポートに失敗しました: {e}")
            
            with col_exp2:
                if st.button("📥 JSON形式でエクスポート"):
                    try:
                        import json
                        export_json = json.dumps(stats, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📥 JSONダウンロード",
                            data=export_json,
                            file_name=f"keyword_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"[NG] エクスポートに失敗しました: {e}")
        except Exception as e:
            st.warning(f"[WARN] キーワード状態の読み込みに失敗しました: {e}")
        
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
        st.subheader("[STATS] 全体統計")
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
                    st.success(f"[OK] [{decision_type}] {keyword} - {decision_result}")
                elif decision_result == 'deny':
                    st.error(f"[NG] [{decision_type}] {keyword} - {decision_result}")
                elif decision_result == 'modify':
                    st.warning(f"[WARN] [{decision_type}] {keyword} - {decision_result}")
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
        st.subheader("[NOTE] 最新サンプル")
        
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

