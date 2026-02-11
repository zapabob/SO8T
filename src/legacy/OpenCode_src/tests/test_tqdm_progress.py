#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T tqdm風プログレスバー テストスクリプト
"""

import time
from tqdm import tqdm

class TimeFormatter:
    """tqdm風の時間フォーマット"""

    @staticmethod
    def format_time(seconds):
        """時間を読みやすいフォーマットに変換"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        elif minutes > 0:
            return f"{minutes:02d}:{seconds:02d}"
        else:
            return f"{seconds:02d}s"

def simulate_training(max_steps=50):
    """トレーニングをシミュレートしてtqdmプログレスバーをテスト"""

    print("SO8T PPO Training Progress Test")
    print("=" * 50)

    start_time = time.time()

    # tqdm風プログレスバー
    progress_bar = tqdm(
        total=max_steps,
        desc="SO8T PPO Training",
        unit="step",
        ncols=120,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for step in range(max_steps):
        # トレーニング処理のシミュレーション
        time.sleep(0.1)  # 0.1秒待機

        # 経過時間と残り時間の計算
        elapsed_time = time.time() - start_time
        if step > 0:
            avg_time_per_step = elapsed_time / (step + 1)
            remaining_steps = max_steps - (step + 1)
            estimated_remaining = avg_time_per_step * remaining_steps

            # 時間フォーマット
            elapsed_str = TimeFormatter.format_time(elapsed_time)
            remaining_str = TimeFormatter.format_time(estimated_remaining)

            # 追加情報をプログレスバーに表示
            progress_bar.set_postfix({
                'loss': f"{0.5 - step * 0.01:.4f}",
                'reward': f"{step * 0.02:.4f}",
                'epoch': f"{step // 10 + 1}/5",
                'elapsed': elapsed_str,
                'remaining': remaining_str
            })

        progress_bar.update(1)

    progress_bar.close()

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f} seconds")
def main():
    """メイン実行関数"""
    simulate_training(30)  # 30ステップでテスト

if __name__ == "__main__":
    main()
