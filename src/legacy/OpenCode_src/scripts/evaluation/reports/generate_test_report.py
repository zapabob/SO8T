#!/usr/bin/env python3
"""
SO8Tテストレポート生成スクリプト

テスト結果を分析してHTMLレポートを生成:
- テスト結果の可視化
- パフォーマンスメトリクス
- エラーログの分析
- トレンド分析
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# データ分析ライブラリ
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("[WARNING] データ分析ライブラリが見つかりません。pip install pandas matplotlib seaborn plotly を実行してください。")
    pd = None
    plt = None
    sns = None
    go = None

logger = logging.getLogger(__name__)


class SO8TTestReportGenerator:
    """SO8Tテストレポート生成クラス"""
    
    def __init__(self, log_dir: str, timestamp: str):
        """
        Args:
            log_dir: ログディレクトリ
            timestamp: タイムスタンプ
        """
        self.log_dir = Path(log_dir)
        self.timestamp = timestamp
        self.report_dir = self.log_dir / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        # レポートデータ
        self.test_data = {}
        self.performance_metrics = {}
        self.error_analysis = {}
        
    def load_test_data(self) -> None:
        """テストデータの読み込み"""
        print("[REPORT] テストデータ読み込み中...")
        
        # 最終結果ファイルの読み込み
        final_results_file = self.log_dir / f"final_results_{self.timestamp}.json"
        if final_results_file.exists():
            with open(final_results_file, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
        
        # 個別テスト結果の読み込み
        test_files = list(self.log_dir.glob("comprehensive_test_results_*.json"))
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'test_data' in data:
                        self.test_data.update(data['test_data'])
            except Exception as e:
                logger.warning(f"テストファイルの読み込みに失敗: {test_file} - {e}")
        
        print(f"[OK] テストデータ読み込み完了: {len(self.test_data)} 項目")
    
    def analyze_performance_metrics(self) -> None:
        """パフォーマンスメトリクスの分析"""
        print("[REPORT] パフォーマンスメトリクス分析中...")
        
        # 基本的なメトリクス
        self.performance_metrics = {
            'total_tests': self.test_data.get('total_tests', 0),
            'successful_tests': self.test_data.get('successful_tests', 0),
            'failed_tests': self.test_data.get('failed_tests', 0),
            'success_rate': self.test_data.get('success_rate', 0),
            'timestamp': self.test_data.get('timestamp', self.timestamp)
        }
        
        # テストカテゴリ別の分析
        if 'test_results' in self.test_data:
            test_results = self.test_data['test_results']
            self.performance_metrics['category_breakdown'] = {}
            
            for test_name, result in test_results.items():
                category = self._get_test_category(test_name)
                if category not in self.performance_metrics['category_breakdown']:
                    self.performance_metrics['category_breakdown'][category] = {
                        'total': 0,
                        'successful': 0,
                        'failed': 0
                    }
                
                self.performance_metrics['category_breakdown'][category]['total'] += 1
                if result:
                    self.performance_metrics['category_breakdown'][category]['successful'] += 1
                else:
                    self.performance_metrics['category_breakdown'][category]['failed'] += 1
        
        print("[OK] パフォーマンスメトリクス分析完了")
    
    def _get_test_category(self, test_name: str) -> str:
        """テスト名からカテゴリを取得"""
        if 'SO8' in test_name or 'Operations' in test_name:
            return 'SO8_Operations'
        elif 'PyTorch' in test_name or 'Comparison' in test_name:
            return 'PyTorch_Comparison'
        elif 'Quantization' in test_name:
            return 'Quantization'
        elif 'Existing' in test_name or 'Integration' in test_name:
            return 'Integration'
        else:
            return 'Other'
    
    def analyze_errors(self) -> None:
        """エラー分析"""
        print("[REPORT] エラー分析中...")
        
        # ログファイルからエラーを抽出
        log_files = list(self.log_dir.glob("comprehensive_test_log_*.log"))
        errors = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # エラーパターンの検索
                    error_patterns = [
                        r'ERROR: (.+)',
                        r'FAILED (.+)',
                        r'Exception: (.+)',
                        r'Traceback \(most recent call last\):',
                        r'AssertionError: (.+)'
                    ]
                    
                    import re
                    for pattern in error_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        errors.extend(matches)
                        
            except Exception as e:
                logger.warning(f"ログファイルの読み込みに失敗: {log_file} - {e}")
        
        self.error_analysis = {
            'total_errors': len(errors),
            'error_types': self._categorize_errors(errors),
            'common_errors': self._find_common_errors(errors)
        }
        
        print(f"[OK] エラー分析完了: {len(errors)} 個のエラーを検出")
    
    def _categorize_errors(self, errors: List[str]) -> Dict[str, int]:
        """エラーのカテゴリ化"""
        categories = {
            'AssertionError': 0,
            'ImportError': 0,
            'RuntimeError': 0,
            'ValueError': 0,
            'TypeError': 0,
            'FileNotFoundError': 0,
            'Other': 0
        }
        
        for error in errors:
            error_lower = error.lower()
            if 'assertion' in error_lower:
                categories['AssertionError'] += 1
            elif 'import' in error_lower:
                categories['ImportError'] += 1
            elif 'runtime' in error_lower:
                categories['RuntimeError'] += 1
            elif 'value' in error_lower:
                categories['ValueError'] += 1
            elif 'type' in error_lower:
                categories['TypeError'] += 1
            elif 'file not found' in error_lower or 'no such file' in error_lower:
                categories['FileNotFoundError'] += 1
            else:
                categories['Other'] += 1
        
        return categories
    
    def _find_common_errors(self, errors: List[str]) -> List[Dict[str, Any]]:
        """共通エラーの検出"""
        from collections import Counter
        
        # エラーメッセージの頻度カウント
        error_counts = Counter(errors)
        common_errors = []
        
        for error, count in error_counts.most_common(10):
            if count > 1:  # 2回以上出現したエラーのみ
                common_errors.append({
                    'error': error,
                    'count': count,
                    'percentage': (count / len(errors)) * 100
                })
        
        return common_errors
    
    def generate_html_report(self) -> str:
        """HTMLレポートの生成"""
        print("[REPORT] HTMLレポート生成中...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SO8Tテストレポート - {self.timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
        }}
        .header p {{
            color: #666;
            margin: 10px 0 0 0;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .metric-card p {{
            margin: 0;
            opacity: 0.9;
        }}
        .success {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }}
        .warning {{
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        }}
        .error {{
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #333;
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .test-results {{
            overflow-x: auto;
        }}
        .test-results table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .test-results th,
        .test-results td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .test-results th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .status-success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-failed {{
            color: #f44336;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SO8Tテストレポート</h1>
            <p>生成時刻: {self.timestamp}</p>
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <h3>{self.performance_metrics.get('total_tests', 0)}</h3>
                <p>総テスト数</p>
            </div>
            <div class="metric-card success">
                <h3>{self.performance_metrics.get('successful_tests', 0)}</h3>
                <p>成功テスト数</p>
            </div>
            <div class="metric-card {'error' if self.performance_metrics.get('failed_tests', 0) > 0 else 'success'}">
                <h3>{self.performance_metrics.get('failed_tests', 0)}</h3>
                <p>失敗テスト数</p>
            </div>
            <div class="metric-card {'warning' if self.performance_metrics.get('success_rate', 0) < 100 else 'success'}">
                <h3>{self.performance_metrics.get('success_rate', 0):.1f}%</h3>
                <p>成功率</p>
            </div>
        </div>
        
        <div class="section">
            <h2>テスト結果詳細</h2>
            <div class="test-results">
                <table>
                    <thead>
                        <tr>
                            <th>テスト名</th>
                            <th>カテゴリ</th>
                            <th>ステータス</th>
                            <th>実行時刻</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        # テスト結果の追加
        if 'test_results' in self.test_data:
            for test_name, result in self.test_data['test_results'].items():
                category = self._get_test_category(test_name)
                status = "SUCCESS" if result else "FAILED"
                status_class = "status-success" if result else "status-failed"
                
                html_content += f"""
                        <tr>
                            <td>{test_name}</td>
                            <td>{category}</td>
                            <td class="{status_class}">{status}</td>
                            <td>{self.timestamp}</td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>カテゴリ別結果</h2>
            <div class="chart-container">
"""
        
        # カテゴリ別結果の追加
        if 'category_breakdown' in self.performance_metrics:
            for category, data in self.performance_metrics['category_breakdown'].items():
                success_rate = (data['successful'] / data['total']) * 100 if data['total'] > 0 else 0
                html_content += f"""
                <div style="margin-bottom: 15px;">
                    <h4>{category}</h4>
                    <p>成功: {data['successful']} / {data['total']} ({success_rate:.1f}%)</p>
                    <div style="background-color: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="background-color: #4CAF50; height: 100%; width: {success_rate}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>エラー分析</h2>
            <div class="chart-container">
"""
        
        # エラー分析の追加
        if self.error_analysis:
            html_content += f"""
                <h4>エラー統計</h4>
                <p>総エラー数: {self.error_analysis.get('total_errors', 0)}</p>
                
                <h4>エラータイプ別分布</h4>
"""
            for error_type, count in self.error_analysis.get('error_types', {}).items():
                if count > 0:
                    html_content += f"<p>{error_type}: {count}</p>"
            
            if self.error_analysis.get('common_errors'):
                html_content += "<h4>共通エラー</h4>"
                for error in self.error_analysis['common_errors'][:5]:
                    html_content += f"<p>{error['error']} (出現回数: {error['count']}, {error['percentage']:.1f}%)</p>"
        
        html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>SO8Tテストレポート生成システム v1.0</p>
            <p>生成時刻: {}</p>
        </div>
    </div>
</body>
</html>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # HTMLファイルの保存
        html_file = self.report_dir / f"test_report_{self.timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[OK] HTMLレポート生成完了: {html_file}")
        return str(html_file)
    
    def generate_json_report(self) -> str:
        """JSONレポートの生成"""
        print("[REPORT] JSONレポート生成中...")
        
        report_data = {
            'timestamp': self.timestamp,
            'generation_time': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'error_analysis': self.error_analysis,
            'test_data': self.test_data,
            'summary': {
                'total_tests': self.performance_metrics.get('total_tests', 0),
                'success_rate': self.performance_metrics.get('success_rate', 0),
                'status': 'SUCCESS' if self.performance_metrics.get('failed_tests', 0) == 0 else 'WARNING'
            }
        }
        
        json_file = self.report_dir / f"test_report_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] JSONレポート生成完了: {json_file}")
        return str(json_file)
    
    def generate_visualizations(self) -> List[str]:
        """可視化の生成"""
        if not pd or not plt or not go:
            print("[WARNING] 可視化ライブラリが利用できません。スキップします。")
            return []
        
        print("[REPORT] 可視化生成中...")
        
        visualizations = []
        
        try:
            # テスト結果の可視化
            if 'category_breakdown' in self.performance_metrics:
                # カテゴリ別成功率の棒グラフ
                categories = list(self.performance_metrics['category_breakdown'].keys())
                success_rates = []
                
                for category in categories:
                    data = self.performance_metrics['category_breakdown'][category]
                    rate = (data['successful'] / data['total']) * 100 if data['total'] > 0 else 0
                    success_rates.append(rate)
                
                fig = go.Figure(data=[
                    go.Bar(x=categories, y=success_rates, marker_color='lightblue')
                ])
                
                fig.update_layout(
                    title='カテゴリ別テスト成功率',
                    xaxis_title='カテゴリ',
                    yaxis_title='成功率 (%)',
                    yaxis=dict(range=[0, 100])
                )
                
                chart_file = self.report_dir / f"category_success_rate_{self.timestamp}.html"
                fig.write_html(str(chart_file))
                visualizations.append(str(chart_file))
            
            # エラータイプの円グラフ
            if self.error_analysis.get('error_types'):
                error_types = list(self.error_analysis['error_types'].keys())
                error_counts = list(self.error_analysis['error_types'].values())
                
                # 0でないエラータイプのみを表示
                non_zero_types = []
                non_zero_counts = []
                for i, count in enumerate(error_counts):
                    if count > 0:
                        non_zero_types.append(error_types[i])
                        non_zero_counts.append(count)
                
                if non_zero_types:
                    fig = go.Figure(data=[
                        go.Pie(labels=non_zero_types, values=non_zero_counts)
                    ])
                    
                    fig.update_layout(
                        title='エラータイプ別分布'
                    )
                    
                    chart_file = self.report_dir / f"error_types_{self.timestamp}.html"
                    fig.write_html(str(chart_file))
                    visualizations.append(str(chart_file))
            
            print(f"[OK] 可視化生成完了: {len(visualizations)} 個のチャート")
            
        except Exception as e:
            logger.error(f"可視化生成中にエラーが発生しました: {e}")
        
        return visualizations
    
    def generate_report(self) -> Dict[str, str]:
        """レポートの生成"""
        print("=" * 80)
        print("SO8Tテストレポート生成開始")
        print("=" * 80)
        
        # データの読み込みと分析
        self.load_test_data()
        self.analyze_performance_metrics()
        self.analyze_errors()
        
        # レポートの生成
        html_file = self.generate_html_report()
        json_file = self.generate_json_report()
        chart_files = self.generate_visualizations()
        
        # 結果のサマリー
        print("\n" + "=" * 80)
        print("レポート生成完了")
        print("=" * 80)
        print(f"HTMLレポート: {html_file}")
        print(f"JSONレポート: {json_file}")
        print(f"チャートファイル: {len(chart_files)} 個")
        
        return {
            'html_report': html_file,
            'json_report': json_file,
            'chart_files': chart_files
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='SO8Tテストレポート生成')
    parser.add_argument('--timestamp', required=True, help='タイムスタンプ')
    parser.add_argument('--log-dir', default='_docs/test_logs', help='ログディレクトリ')
    parser.add_argument('--output-dir', help='出力ディレクトリ（指定しない場合はlog-dir内にreportsディレクトリを作成）')
    
    args = parser.parse_args()
    
    # レポート生成器の作成
    generator = SO8TTestReportGenerator(args.log_dir, args.timestamp)
    
    # レポートの生成
    try:
        results = generator.generate_report()
        print("\n[SUCCESS] レポート生成が正常に完了しました")
        return 0
    except Exception as e:
        print(f"\n[ERROR] レポート生成中にエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
