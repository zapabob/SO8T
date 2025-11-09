#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統合Phi-3モデル評価スクリプト

基本評価メトリクス:
- Perplexity計算
- 損失値の評価
- 生成品質の評価

SO8T固有評価:
- 直交性正則化損失の確認
- SO(8)群構造の維持確認
- 回転ゲートの動作確認
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SO8TModelEvaluator:
    """SO8T統合モデル評価器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: モデルパス
            device: デバイス
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        logger.info("="*80)
        logger.info("SO8T Model Evaluator")
        logger.info("="*80)
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {self.device}")
        
        # モデルとトークナイザー読み込み
        self._load_model()
    
    def _load_model(self):
        """モデルとトークナイザーを読み込み"""
        logger.info("Loading model and tokenizer...")
        
        try:
            self.config = AutoConfig.from_pretrained(str(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # モデル読み込み（SO8T統合済みモデル）
            # 注意: modeling_phi3_so8t.pyを使用する必要がある
            try:
                # カスタムモデルクラスをインポート
                import importlib.util
                modeling_path = self.model_path / "modeling_phi3_so8t.py"
                if modeling_path.exists():
                    spec = importlib.util.spec_from_file_location("modeling_phi3_so8t", modeling_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # カスタムモデルクラスを使用
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        config=self.config,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    ).to(self.device)
                else:
                    # 標準モデルを使用
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(self.model_path),
                        config=self.config,
                        torch_dtype=torch.bfloat16
                    ).to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load custom model, using standard: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    config=self.config,
                    torch_dtype=torch.bfloat16
                ).to(self.device)
            
            self.model.eval()
            logger.info("[OK] Model and tokenizer loaded")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> float:
        """Perplexityを計算"""
        logger.info("Calculating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
                batch_texts = texts[i:i+batch_size]
                
                # トークナイズ
                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask
                
                # 損失計算
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                num_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Perplexity: {perplexity:.4f}")
        
        return perplexity
    
    def evaluate_so8t_components(self) -> Dict[str, float]:
        """SO8T固有コンポーネントの評価"""
        logger.info("Evaluating SO8T components...")
        
        results = {}
        
        try:
            from models.so8t_rotation_gate import SO8TRotationGate
            
            # SO8TRotationGateの検索
            so8t_gates = []
            for name, module in self.model.named_modules():
                if isinstance(module, SO8TRotationGate):
                    so8t_gates.append((name, module))
            
            if not so8t_gates:
                logger.warning("No SO8TRotationGate found in model")
                return results
            
            logger.info(f"Found {len(so8t_gates)} SO8TRotationGate instances")
            
            # 直交性損失の計算
            total_ortho_loss = 0.0
            for name, gate in so8t_gates:
                ortho_loss = gate.get_orthogonality_loss()
                total_ortho_loss += ortho_loss.item()
                logger.info(f"  {name}: orthogonality_loss = {ortho_loss.item():.6f}")
            
            avg_ortho_loss = total_ortho_loss / len(so8t_gates) if so8t_gates else 0.0
            results['avg_orthogonality_loss'] = avg_ortho_loss
            results['num_so8t_gates'] = len(so8t_gates)
            
            logger.info(f"[OK] Average orthogonality loss: {avg_ortho_loss:.6f}")
            
        except ImportError:
            logger.warning("SO8TRotationGate not available")
        except Exception as e:
            logger.error(f"Failed to evaluate SO8T components: {e}")
        
        return results
    
    def evaluate_generation_quality(self, prompts: List[str], max_length: int = 100) -> Dict[str, float]:
        """生成品質を評価"""
        logger.info("Evaluating generation quality...")
        
        results = {
            'avg_length': 0.0,
            'num_generated': 0
        }
        
        total_length = 0
        generated_count = 0
        
        with torch.no_grad():
            for prompt in tqdm(prompts[:10], desc="Generation"):  # 最大10件
                try:
                    # トークナイズ
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    # 生成
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # デコード
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_length = len(generated_text)
                    
                    total_length += generated_length
                    generated_count += 1
                    
                except Exception as e:
                    logger.debug(f"Generation failed: {e}")
                    continue
        
        if generated_count > 0:
            results['avg_length'] = total_length / generated_count
            results['num_generated'] = generated_count
        
        logger.info(f"[OK] Generated {generated_count} samples, avg length: {results['avg_length']:.1f}")
        
        return results
    
    def evaluate(self, eval_dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """評価を実行"""
        logger.info("="*80)
        logger.info("Running Evaluation")
        logger.info("="*80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'device': str(self.device)
        }
        
        # 評価データセット読み込み
        texts = []
        if eval_dataset_path and Path(eval_dataset_path).exists():
            logger.info(f"Loading evaluation dataset: {eval_dataset_path}")
            with open(eval_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('content', data.get('text', ''))
                        if text:
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(texts)} evaluation samples")
        else:
            # サンプルテキストを使用
            logger.warning("No evaluation dataset provided, using sample texts")
            texts = [
                "人工知能は、コンピュータシステムが人間の知能を模倣する技術です。",
                "機械学習はAIの一分野で、データから学習してパターンを認識します。",
                "深層学習は機械学習の一種で、ニューラルネットワークを使用します。"
            ]
        
        # Perplexity計算
        if texts:
            perplexity = self.calculate_perplexity(texts)
            results['perplexity'] = perplexity
        
        # SO8T固有評価
        so8t_results = self.evaluate_so8t_components()
        results['so8t_components'] = so8t_results
        
        # 生成品質評価
        prompts = ["人工知能について説明してください。", "機械学習とは何ですか？"]
        generation_results = self.evaluate_generation_quality(prompts)
        results['generation_quality'] = generation_results
        
        logger.info("="*80)
        logger.info("Evaluation Results")
        logger.info("="*80)
        logger.info(json.dumps(results, indent=2, ensure_ascii=False))
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """結果を保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Markdown形式でも保存
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# SO8T Model Evaluation Results\n\n")
            f.write(f"**Date**: {results['timestamp']}\n\n")
            f.write(f"**Model Path**: {results['model_path']}\n\n")
            f.write(f"**Device**: {results['device']}\n\n")
            
            if 'perplexity' in results:
                f.write("## Perplexity\n\n")
                f.write(f"**Perplexity**: {results['perplexity']:.4f}\n\n")
            
            if 'so8t_components' in results:
                f.write("## SO8T Components\n\n")
                for key, value in results['so8t_components'].items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            if 'generation_quality' in results:
                f.write("## Generation Quality\n\n")
                for key, value in results['generation_quality'].items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
        
        logger.info(f"[SAVE] Results saved to {output_path}")
        logger.info(f"[SAVE] Markdown report saved to {md_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='SO8T Model Evaluator')
    parser.add_argument('--model-path', type=str, required=True, help='Model path')
    parser.add_argument('--eval-dataset', type=str, default=None, help='Evaluation dataset path (JSONL)')
    parser.add_argument('--output', type=str, default='D:/webdataset/evaluation_results/evaluation.json')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    evaluator = SO8TModelEvaluator(args.model_path, device=args.device)
    results = evaluator.evaluate(eval_dataset_path=args.eval_dataset)
    evaluator.save_results(results, Path(args.output))


if __name__ == '__main__':
    main()

