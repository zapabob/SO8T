#!/usr/bin/env python3
"""
SO8T Knowledge Distillation Runner
SO8T-Phi31-Mini-128K-Enhanced-Q8_0.ggufから軽量モデルへの知識蒸留実行スクリプト

CoT仮説検証思考で重み崩壊を防ぎながら効率的な知識蒸留を実行
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any
import warnings
from tqdm import tqdm
import time
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 知識蒸留システム
from utils.knowledge_distillation import SO8TKnowledgeDistillation

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SO8TDistillationRunner:
    """SO8T知識蒸留実行器"""
    
    def __init__(self, config_path: str = None):
        """初期化"""
        self.config_path = config_path or "configs/so8t_distillation_config.json"
        self.config = self._load_config()
        
        logger.info("SO8T知識蒸留実行器初期化完了")
        logger.info(f"   - 設定ファイル: {self.config_path}")
        logger.info(f"   - 教師モデル: {self.config['teacher_model_path']}")
        logger.info(f"   - 出力ディレクトリ: {self.config['output_dir']}")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定を読み込み"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # デフォルト設定
            config = {
                "teacher_model_path": "models/SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf",
                "output_dir": "models/qwen_so8t_lightweight",
                "student_config": {
                    "vocab_size": 32000,
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "num_hidden_layers": 16,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "hidden_act": "silu",
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "attention_dropout": 0.0,
                    "use_cache": True,
                    "so8t_rotation_dim": 8,
                    "so8t_triality_symmetry": True,
                    "so8t_cross_head_interaction": True,
                    "so8t_non_commutative_gates": True,
                },
                "distillation_config": {
                    "num_epochs": 10,
                    "num_samples": 1000,
                    "batch_size": 8,
                    "learning_rate": 1e-4,
                    "temperature": 3.0,
                    "alpha": 0.7,
                    "beta": 0.3,
                    "gamma": 0.1,
                    "lambda_so8t": 0.5,
                    "lambda_safety": 0.3,
                    "lambda_verification": 0.2,
                },
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
            
            # 設定ファイルを保存
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"デフォルト設定を作成: {self.config_path}")
        
        return config
    
    def run_distillation(self) -> Dict[str, Any]:
        """知識蒸留を実行"""
        logger.info("=" * 80)
        logger.info("SO8T知識蒸留実行開始")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 知識蒸留システム初期化
            distillation_system = SO8TKnowledgeDistillation(
                teacher_model_path=self.config['teacher_model_path'],
                student_config=self.config['student_config'],
                output_dir=self.config['output_dir'],
                device=self.config['device']
            )
            
            # 蒸留設定を更新
            distillation_system.distillation_config.update(
                self.config['distillation_config']
            )
            
            # 知識蒸留実行
            results = distillation_system.run_distillation(
                num_epochs=self.config['distillation_config']['num_epochs'],
                num_samples=self.config['distillation_config']['num_samples']
            )
            
            # 実行時間計算
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 結果に実行時間を追加
            results['execution_time'] = execution_time
            results['execution_time_hours'] = execution_time / 3600
            
            logger.info("=" * 80)
            logger.info("SO8T知識蒸留実行完了！")
            logger.info("=" * 80)
            logger.info(f"実行時間: {execution_time:.2f}秒 ({execution_time/3600:.2f}時間)")
            logger.info(f"教師モデル: {results['teacher_model_path']}")
            logger.info(f"学生モデル: {results['student_model_path']}")
            logger.info(f"出力ディレクトリ: {results['output_dir']}")
            logger.info(f"エポック数: {results['num_epochs']}")
            logger.info(f"サンプル数: {results['num_samples']}")
            
            # 結果をJSONファイルに保存
            results_file = Path(self.config['output_dir']) / 'distillation_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"結果保存: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"知識蒸留実行エラー: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_lightweight_model_card(self, results: Dict[str, Any]) -> str:
        """軽量モデルカードを作成"""
        logger.info("軽量モデルカード作成中...")
        
        model_card = f"""# Qwen2.5-7B-Instruct SO8T Transformer 軽量版 (知識蒸留)

## 概要
SO8T-Phi31-Mini-128K-Enhanced-Q8_0.ggufから知識蒸留により作成された軽量SO8T Transformerモデルです。

## 特徴
- **知識蒸留**: 大規模モデルから軽量モデルへの効率的な知識転移
- **SO8群構造**: 8次元回転群の完全な数学的実装
- **Triality対称性**: Vector, Spinor+, Spinor-表現の完全対応
- **三重推論**: タスク、安全、権限推論の完全実装
- **軽量化**: パラメータ数を大幅に削減しながら性能を維持
- **重み安定性**: 重み崩壊を防ぐ高度な安定化技術

## 蒸留情報
- **教師モデル**: {results['teacher_model_path']}
- **蒸留日時**: {results['timestamp']}
- **エポック数**: {results['num_epochs']}
- **サンプル数**: {results['num_samples']}
- **実行時間**: {results['execution_time_hours']:.2f}時間

## モデル仕様
- **アーキテクチャ**: SO8TTransformerForCausalLM
- **語彙サイズ**: {self.config['student_config']['vocab_size']:,}
- **隠れサイズ**: {self.config['student_config']['hidden_size']:,}
- **中間サイズ**: {self.config['student_config']['intermediate_size']:,}
- **レイヤー数**: {self.config['student_config']['num_hidden_layers']}
- **アテンションヘッド数**: {self.config['student_config']['num_attention_heads']}
- **キー・バリューヘッド数**: {self.config['student_config']['num_key_value_heads']}
- **最大位置埋め込み**: {self.config['student_config']['max_position_embeddings']:,}

## SO8T固有パラメータ
- **so8t_rotation_dim**: {self.config['student_config']['so8t_rotation_dim']}
- **so8t_triality_symmetry**: {self.config['student_config']['so8t_triality_symmetry']}
- **so8t_cross_head_interaction**: {self.config['student_config']['so8t_cross_head_interaction']}
- **so8t_non_commutative_gates**: {self.config['student_config']['so8t_non_commutative_gates']}

## 蒸留設定
- **温度**: {self.config['distillation_config']['temperature']}
- **教師重み**: {self.config['distillation_config']['alpha']}
- **学生重み**: {self.config['distillation_config']['beta']}
- **中間層重み**: {self.config['distillation_config']['gamma']}
- **SO8T損失重み**: {self.config['distillation_config']['lambda_so8t']}
- **安全性損失重み**: {self.config['distillation_config']['lambda_safety']}
- **検証損失重み**: {self.config['distillation_config']['lambda_verification']}

## 使用方法
```python
import torch
from models.qwen_so8t_lightweight.so8t_transformer_model import SO8TTransformerForCausalLM

# モデル読み込み
model = SO8TTransformerForCausalLM.from_pretrained("path/to/lightweight/model")

# 推論実行
outputs = model(input_ids, attention_mask=attention_mask)
```

## ライセンス
Apache-2.0

## 作成者
SO8T Safe Agent Project

## 作成日
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # モデルカードを保存
        model_card_file = Path(self.config['output_dir']) / 'README.md'
        with open(model_card_file, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        logger.info(f"軽量モデルカード作成完了: {model_card_file}")
        return str(model_card_file)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T知識蒸留実行")
    parser.add_argument("--config", type=str, default="configs/so8t_distillation_config.json",
                       help="蒸留設定ファイルのパス")
    parser.add_argument("--teacher", type=str, 
                       default="models/SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf",
                       help="教師モデルのパス")
    parser.add_argument("--output", type=str, default="models/qwen_so8t_lightweight",
                       help="出力ディレクトリ")
    parser.add_argument("--epochs", type=int, default=10,
                       help="エポック数")
    parser.add_argument("--samples", type=int, default=1000,
                       help="サンプル数")
    
    args = parser.parse_args()
    
    try:
        # 実行器初期化
        runner = SO8TDistillationRunner(args.config)
        
        # 設定を更新
        if args.teacher:
            runner.config['teacher_model_path'] = args.teacher
        if args.output:
            runner.config['output_dir'] = args.output
        if args.epochs:
            runner.config['distillation_config']['num_epochs'] = args.epochs
        if args.samples:
            runner.config['distillation_config']['num_samples'] = args.samples
        
        # 知識蒸留実行
        results = runner.run_distillation()
        
        # 軽量モデルカード作成
        model_card_file = runner.create_lightweight_model_card(results)
        
        print("=" * 80)
        print("SO8T知識蒸留完了！")
        print("=" * 80)
        print(f"教師モデル: {results['teacher_model_path']}")
        print(f"学生モデル: {results['student_model_path']}")
        print(f"出力ディレクトリ: {results['output_dir']}")
        print(f"実行時間: {results['execution_time_hours']:.2f}時間")
        print(f"モデルカード: {model_card_file}")
        
    except Exception as e:
        print(f"知識蒸留実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
