#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統制Webスクレイピング集中管理ダッシュボード

Streamlitを使用して、進行状況と各ブラウザを集中管理するダッシュボード

Usage:
    streamlit run scripts/dashboard/so8t_scraping_dashboard.py
"""

import sys
import json
import time
import asyncio
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


class ScrapingDashboard:
    """SO8T統制Webスクレイピングダッシュボード"""
    
    def __init__(self):
        """初期化"""
        self.output_dir = Path("D:/webdataset/processed")
        self.log_dir = Path("logs")
        self.checkpoint_dir = Path("D:/webdataset/checkpoints/pipeline")
        
        # セッション状態の初期化
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'scraping_stats' not in st.session_state:
            st.session_state.scraping_stats = {
                'total_samples': 0,
                'nsfw_samples': 0,
                'processed_keywords': 0,
                'total_keywords': 0,
                'browser_status': {}
            }
    
    def load_dashboard_state(self) -> Optional[Dict]:
        """ダッシュボード状態を読み込み"""
        state_file = self.output_dir / "dashboard_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load dashboard state: {e}")
        
        return None
    
    def load_latest_samples(self) -> List[Dict]:
        """最新のサンプルを読み込み"""
        samples = []
        if self.output_dir.exists():
            # 最新のJSONLファイルを探す
            jsonl_files = sorted(
                self.output_dir.glob("parallel_deep_research_scraped_*.jsonl"),
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
    
    def load_log_data(self) -> List[str]:
        """ログデータを読み込み"""
        log_lines = []
        log_file = self.log_dir / "parallel_deep_research_scraping.log"
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    # 最後の1000行を読み込み
                    lines = f.readlines()
                    log_lines = lines[-1000:] if len(lines) > 1000 else lines
            except Exception as e:
                logger.error(f"Failed to load log: {e}")
        
        return log_lines
    
    def parse_browser_status_from_logs(self, log_lines: List[str]) -> Dict[int, Dict]:
        """ログからブラウザ状態を解析"""
        browser_status = {}
        
        for line in log_lines:
            if "[BROWSER" in line and "]" in line:
                # ブラウザ番号を抽出
                try:
                    browser_num = int(line.split("[BROWSER")[1].split("]")[0].strip())
                    
                    if browser_num not in browser_status:
                        browser_status[browser_num] = {
                            'status': 'active',
                            'current_keyword': None,
                            'samples_collected': 0,
                            'last_activity': None
                        }
                    
                    # キーワード処理中
                    if "Processing keyword:" in line:
                        keyword = line.split("Processing keyword:")[1].strip()
                        browser_status[browser_num]['current_keyword'] = keyword
                        browser_status[browser_num]['last_activity'] = datetime.now().isoformat()
                    
                    # サンプル収集
                    if "Collected" in line and "samples" in line:
                        try:
                            count = int(line.split("Collected")[1].split("samples")[0].strip())
                            browser_status[browser_num]['samples_collected'] += count
                        except:
                            pass
                    
                    # 完了
                    if "finished" in line.lower() or "completed" in line.lower():
                        browser_status[browser_num]['status'] = 'completed'
                    
                except Exception:
                    continue
        
        return browser_status
    
    def parse_so8t_decisions_from_logs(self, log_lines: List[str]) -> List[Dict]:
        """ログからSO8T判断結果を解析"""
        decisions = []
        
        for line in log_lines:
            if "[SO8T]" in line:
                decision = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'unknown',
                    'decision': 'unknown',
                    'reasoning': ''
                }
                
                if "Search denied" in line or "Search modified" in line:
                    decision['type'] = 'search'
                    decision['decision'] = 'denied' if "denied" in line else 'modified'
                elif "Scraping denied" in line or "Scraping modified" in line:
                    decision['type'] = 'scrape'
                    decision['decision'] = 'denied' if "denied" in line else 'modified'
                elif "Bypass denied" in line or "Bypass modified" in line:
                    decision['type'] = 'bypass'
                    decision['decision'] = 'denied' if "denied" in line else 'modified'
                
                if "Reasoning:" in line:
                    decision['reasoning'] = line.split("Reasoning:")[1].strip()
                
                decisions.append(decision)
        
        return decisions[-50:]  # 最後の50件
    
    def load_browser_screenshots(self, screenshots_dir: Path, browser_status: Dict[int, Dict]) -> List[tuple]:
        """ブラウザスクリーンショットを読み込み"""
        screenshots = []
        
        if not screenshots_dir.exists():
            return screenshots
        
        # ブラウザ番号ごとに最新のスクリーンショットを取得
        for browser_num in sorted(browser_status.keys()):
            status = browser_status[browser_num]
            
            # スクリーンショットパスを取得
            screenshot_path = status.get('screenshot_path')
            if screenshot_path:
                # 相対パスの場合は絶対パスに変換
                if not Path(screenshot_path).is_absolute():
                    screenshot_path = screenshots_dir.parent / screenshot_path
                else:
                    screenshot_path = Path(screenshot_path)
            else:
                # スクリーンショットパスがない場合は、ディレクトリから最新のものを探す
                browser_screenshots = sorted(
                    screenshots_dir.glob(f"browser_{browser_num}_*.png"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if browser_screenshots:
                    screenshot_path = browser_screenshots[0]
                else:
                    screenshot_path = None
            
            screenshot_info = {
                'screenshot_path': str(screenshot_path) if screenshot_path else None,
                'timestamp': status.get('screenshot_timestamp', status.get('last_activity', '不明')),
                'status': status.get('status', 'unknown'),
                'keyword': status.get('current_keyword', None)
            }
            
            screenshots.append((browser_num, screenshot_info))
        
        return screenshots
    
    def calculate_statistics(self, samples: List[Dict]) -> Dict:
        """統計情報を計算"""
        stats = {
            'total_samples': len(samples),
            'nsfw_samples': len([s for s in samples if s.get('nsfw_label') != 'safe']),
            'by_category': {},
            'by_language': {},
            'by_source': {},
            'avg_text_length': 0,
            'total_text_length': 0
        }
        
        if samples:
            # カテゴリ別
            for sample in samples:
                category = sample.get('category', 'unknown')
                stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # 言語別
            for sample in samples:
                language = sample.get('language', 'unknown')
                stats['by_language'][language] = stats['by_language'].get(language, 0) + 1
            
            # ソース別
            for sample in samples:
                source = sample.get('source', 'unknown')
                stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            # 平均テキスト長
            text_lengths = [s.get('text_length', 0) for s in samples if s.get('text_length')]
            if text_lengths:
                stats['total_text_length'] = sum(text_lengths)
                stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
        
        return stats
    
    def render_dashboard(self):
        """ダッシュボードをレンダリング"""
        st.set_page_config(
            page_title="SO8T統制Webスクレイピングダッシュボード",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 SO8T統制Webスクレイピング集中管理ダッシュボード")
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
            # ダッシュボード状態を読み込み（優先）
            dashboard_state = self.load_dashboard_state()
            
            samples = self.load_latest_samples()
            log_lines = self.load_log_data()
            
            # ダッシュボード状態からブラウザ状態とSO8T判断を取得
            if dashboard_state:
                browser_status = dashboard_state.get('browser_status', {})
                so8t_decisions = dashboard_state.get('so8t_decisions', [])
            else:
                # フォールバック: ログから解析
                browser_status = self.parse_browser_status_from_logs(log_lines)
                so8t_decisions = self.parse_so8t_decisions_from_logs(log_lines)
            
            stats = self.calculate_statistics(samples)
            
            # ダッシュボード状態の統計を更新
            if dashboard_state:
                stats['total_samples'] = dashboard_state.get('total_samples', stats['total_samples'])
                stats['nsfw_samples'] = dashboard_state.get('nsfw_samples', stats['nsfw_samples'])
        
        # メトリクス表示
        st.subheader("[STATS] 全体統計")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("総サンプル数", f"{stats['total_samples']:,}")
        with col2:
            st.metric("NSFW検知サンプル", f"{stats['nsfw_samples']:,}", 
                     delta=f"{stats['nsfw_samples']/max(stats['total_samples'], 1)*100:.1f}%" if stats['total_samples'] > 0 else "0%")
        with col3:
            st.metric("平均テキスト長", f"{stats['avg_text_length']:.0f}" if stats['avg_text_length'] > 0 else "0")
        with col4:
            st.metric("総テキスト長", f"{stats['total_text_length']:,}" if stats['total_text_length'] > 0 else "0")
        
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
                    '最終活動': status.get('last_activity', 'なし')
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
        
        # スクリーンショットディレクトリを取得
        screenshots_dir = None
        if dashboard_state:
            screenshots_dir_str = dashboard_state.get('screenshots_dir')
            if screenshots_dir_str:
                screenshots_dir = Path(screenshots_dir_str)
            else:
                screenshots_dir = self.output_dir / "screenshots"
        else:
            screenshots_dir = self.output_dir / "screenshots"
        
        # スクリーンショットを読み込み
        screenshots = self.load_browser_screenshots(screenshots_dir, browser_status)
        
        if screenshots:
            # ブラウザごとにスクリーンショットを表示
            # 2列のグリッドレイアウト
            num_browsers = len(screenshots)
            cols_per_row = 2
            
            for row_start in range(0, num_browsers, cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    browser_idx = row_start + col_idx
                    if browser_idx < num_browsers:
                        browser_num, screenshot_info = screenshots[browser_idx]
                        with cols[col_idx]:
                            st.markdown(f"**Browser {browser_num}**")
                            if screenshot_info['status']:
                                st.markdown(f"*状態: {screenshot_info['status']}*")
                            if screenshot_info['keyword']:
                                st.markdown(f"*キーワード: {screenshot_info['keyword']}*")
                            if screenshot_info['screenshot_path'] and Path(screenshot_info['screenshot_path']).exists():
                                try:
                                    img = Image.open(screenshot_info['screenshot_path'])
                                    st.image(img, use_container_width=True, caption=f"Browser {browser_num} - {screenshot_info['timestamp']}")
                                except Exception as e:
                                    st.error(f"画像読み込みエラー: {e}")
                            else:
                                st.info("スクリーンショットがありません")
        else:
            st.info("スクリーンショットデータが見つかりません。スクレイピングが開始されていない可能性があります。")
        
        st.markdown("---")
        
        # SO8T統制判断結果
        st.subheader("🤖 SO8T統制判断結果")
        
        if so8t_decisions:
            # 判断結果テーブル
            decisions_df_data = []
            for decision in so8t_decisions:
                decisions_df_data.append({
                    'タイムスタンプ': decision.get('timestamp', ''),
                    'タイプ': decision.get('type', 'unknown'),
                    '判断': decision.get('decision', 'unknown'),
                    '推論': decision.get('reasoning', '')[:100] + '...' if len(decision.get('reasoning', '')) > 100 else decision.get('reasoning', '')
                })
            
            decisions_df = pd.DataFrame(decisions_df_data)
            st.dataframe(decisions_df, use_container_width=True)
            
            # 判断結果の可視化
            col1, col2 = st.columns(2)
            
            with col1:
                # タイプ別判断数
                type_counts = {}
                for decision in so8t_decisions:
                    t = decision.get('type', 'unknown')
                    type_counts[t] = type_counts.get(t, 0) + 1
                
                if type_counts:
                    st.bar_chart(type_counts)
            
            with col2:
                # 判断別数
                decision_counts = {}
                for decision in so8t_decisions:
                    d = decision.get('decision', 'unknown')
                    decision_counts[d] = decision_counts.get(d, 0) + 1
                
                if decision_counts:
                    st.bar_chart(decision_counts)
        else:
            st.info("SO8T判断結果データが見つかりません")
        
        st.markdown("---")
        
        # カテゴリ別統計
        st.subheader("📈 カテゴリ別統計")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if stats['by_category']:
                st.bar_chart(stats['by_category'])
        
        with col2:
            if stats['by_language']:
                st.bar_chart(stats['by_language'])
        
        with col3:
            if stats['by_source']:
                st.bar_chart(stats['by_source'])
        
        st.markdown("---")
        
        # 最新サンプル表示
        st.subheader("📄 最新サンプル")
        
        if samples:
            # 最新10件を表示
            recent_samples = samples[-10:]
            
            for i, sample in enumerate(reversed(recent_samples)):
                with st.expander(f"サンプル {len(samples) - i}: {sample.get('keyword', 'unknown')} - {sample.get('url', 'unknown')[:50]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**URL**: {sample.get('url', 'N/A')}")
                        st.write(f"**キーワード**: {sample.get('keyword', 'N/A')}")
                        st.write(f"**カテゴリ**: {sample.get('category', 'N/A')}")
                        st.write(f"**言語**: {sample.get('language', 'N/A')}")
                    
                    with col2:
                        st.write(f"**テキスト長**: {sample.get('text_length', 0):,}")
                        st.write(f"**NSFWラベル**: {sample.get('nsfw_label', 'N/A')}")
                        st.write(f"**NSFW信頼度**: {sample.get('nsfw_confidence', 0):.2f}")
                        st.write(f"**収集時刻**: {sample.get('crawled_at', 'N/A')}")
                    
                    # テキストプレビュー
                    text = sample.get('text', '')
                    if text:
                        st.text_area("テキストプレビュー", text[:500] + "..." if len(text) > 500 else text, height=150, key=f"sample_{i}")
        else:
            st.info("サンプルデータが見つかりません")
        
        st.markdown("---")
        
        # ログ表示
        st.subheader("📋 最新ログ")
        
        if log_lines:
            # 最後の100行を表示
            recent_logs = log_lines[-100:]
            log_text = "\n".join(recent_logs)
            st.text_area("ログ", log_text, height=300)
        else:
            st.info("ログデータが見つかりません")
        
        # フッター
        st.markdown("---")
        st.markdown(f"**最終更新**: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """メイン関数"""
    dashboard = ScrapingDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()

