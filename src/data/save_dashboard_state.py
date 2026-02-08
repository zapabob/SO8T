#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ダッシュボード用状態保存スクリプト

スクレイピング中の状態を定期的にJSONファイルに保存して、
ダッシュボードから読み込めるようにする
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


def save_dashboard_state(
    browser_status: Dict[int, Dict],
    so8t_decisions: List[Dict],
    total_samples: int,
    nsfw_samples: int,
    output_dir: Path
):
    """
    ダッシュボード用状態を保存
    
    Args:
        browser_status: ブラウザ状態辞書
        so8t_decisions: SO8T判断結果リスト
        total_samples: 総サンプル数
        nsfw_samples: NSFWサンプル数
        output_dir: 出力ディレクトリ
    """
    state_file = output_dir / "dashboard_state.json"
    
    # スクリーンショットパスを相対パスに変換（保存用）
    browser_status_for_save = {}
    for browser_index, status in browser_status.items():
        browser_status_for_save[browser_index] = status.copy()
        if 'screenshot_path' in browser_status_for_save[browser_index]:
            # 絶対パスを相対パスに変換
            screenshot_path = Path(browser_status_for_save[browser_index]['screenshot_path'])
            if screenshot_path.is_absolute():
                try:
                    browser_status_for_save[browser_index]['screenshot_path'] = str(screenshot_path.relative_to(output_dir.parent))
                except ValueError:
                    # 相対パスに変換できない場合はそのまま
                    pass
    
    state = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': total_samples,
        'nsfw_samples': nsfw_samples,
        'browser_status': browser_status_for_save,
        'so8t_decisions': so8t_decisions[-100:],  # 最後の100件のみ
        'num_browsers': len(browser_status),
        'screenshots_dir': str(output_dir / "screenshots")
    }
    
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.debug(f"[DASHBOARD] State saved to {state_file}")
    except Exception as e:
        logger.error(f"[DASHBOARD] Failed to save state: {e}")

