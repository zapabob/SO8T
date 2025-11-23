#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuningしたHugging FaceモデルをSO8TTransformerModelに置き換え

Fine-tuningしたモデルをSO8Tで使用可能な形式に変換し、
既存のSO8TTransformerModelと置き換える
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/replace_so8t_with_finetuned_hf.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SO8TModelReplacer:
    """SO8Tモデル置き換えクラス"""
    
    def __init__(
        self,
        finetuned_model_path: Path,
        output_dir: Path,
        so8t_config_path: Optional[Path] = None
    ):
        """
        Args:
            finetuned_model_path: Fine-tuningしたモデルパス
            output_dir: 出力ディレクトリ
            so8t_config_path: SO8T設定ファイルパス（オプション）
        """
        self.finetuned_model_path = Path(finetuned_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.so8t_config_path = so8t_config_path
        
        logger.info("="*80)
        logger.info("SO8T Model Replacer Initialized")
        logger.info("="*80)
        logger.info(f"Finetuned model: {finetuned_model_path}")
        logger.info(f"Output: {output_dir}")
    
    def load_finetuned_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Fine-tuningしたモデルを読み込み"""
        logger.info(f"Loading finetuned model from {self.finetuned_model_path}...")
        
        try:
            # モデル読み込み
            model = AutoModelForCausalLM.from_pretrained(
                str(self.finetuned_model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # トークナイザー読み込み
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.finetuned_model_path),
                trust_remote_code=True
            )
            
            logger.info("[OK] Model and tokenizer loaded")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            raise
    
    def extract_model_weights(self, model: AutoModelForCausalLM) -> Dict[str, torch.Tensor]:
        """モデル重みを抽出"""
        logger.info("Extracting model weights...")
        
        weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights[name] = param.data.clone()
        
        logger.info(f"[OK] Extracted {len(weights)} weight tensors")
        return weights
    
    def convert_to_so8t_format(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer
    ) -> Dict:
        """
        Hugging FaceモデルをSO8T形式に変換
        
        Note: 完全な変換は複雑なため、ここでは重みマッピングとメタデータを保存
        """
        logger.info("Converting to SO8T format...")
        
        # モデル設定取得
        config = model.config
        
        # SO8T形式のメタデータ
        so8t_metadata = {
            "model_type": "hf_finetuned",
            "base_model": str(self.finetuned_model_path),
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": getattr(config, "intermediate_size", config.hidden_size * 4),
            "max_position_embeddings": getattr(config, "max_position_embeddings", 2048),
            "tokenizer_name": tokenizer.name_or_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # 重み抽出
        weights = self.extract_model_weights(model)
        
        # SO8T形式で保存
        so8t_model_path = self.output_dir / "so8t_finetuned_model.pt"
        torch.save({
            "metadata": so8t_metadata,
            "state_dict": weights,
            "config": config.to_dict()
        }, so8t_model_path)
        
        logger.info(f"[OK] SO8T format model saved to {so8t_model_path}")
        
        return {
            "so8t_model_path": so8t_model_path,
            "metadata": so8t_metadata,
            "hf_model_path": str(self.finetuned_model_path)
        }
    
    def create_model_wrapper(self, conversion_result: Dict) -> Path:
        """SO8Tモデルラッパーを作成"""
        logger.info("Creating SO8T model wrapper...")
        
        wrapper_code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Model Wrapper for Fine-tuned Hugging Face Model

This wrapper allows using fine-tuned Hugging Face models with SO8T infrastructure.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Optional, Dict

class SO8TFinetunedModelWrapper(nn.Module):
    """SO8T用Fine-tuning済みモデルラッパー"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        
        # モデル読み込み
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # トークナイザー読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass compatible with SO8TTransformerModel interface"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generate text"""
        return self.model.generate(input_ids, **kwargs)


def load_so8t_finetuned_model(model_path: str, device: str = "auto"):
    """SO8T用Fine-tuning済みモデルを読み込み"""
    return SO8TFinetunedModelWrapper(model_path, device=device)
'''
        
        wrapper_path = self.output_dir / "so8t_finetuned_wrapper.py"
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        logger.info(f"[OK] Wrapper saved to {wrapper_path}")
        
        return wrapper_path
    
    def replace(self) -> Dict:
        """モデル置き換え実行"""
        logger.info("="*80)
        logger.info("Starting Model Replacement")
        logger.info("="*80)
        
        # Fine-tuningしたモデル読み込み
        model, tokenizer = self.load_finetuned_model()
        
        # SO8T形式に変換
        conversion_result = self.convert_to_so8t_format(model, tokenizer)
        
        # ラッパー作成
        wrapper_path = self.create_model_wrapper(conversion_result)
        
        # 置き換え結果
        replacement_result = {
            **conversion_result,
            "wrapper_path": str(wrapper_path),
            "replacement_timestamp": datetime.now().isoformat()
        }
        
        # 結果保存
        result_path = self.output_dir / "replacement_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(replacement_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Replacement result saved to {result_path}")
        
        return replacement_result


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Replace SO8T Model with Fine-tuned Hugging Face Model")
    parser.add_argument(
        '--finetuned-model',
        type=Path,
        required=True,
        help='Fine-tuned model path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--so8t-config',
        type=Path,
        help='SO8T config file path (optional)'
    )
    
    args = parser.parse_args()
    
    # モデル置き換え実行
    replacer = SO8TModelReplacer(
        finetuned_model_path=args.finetuned_model,
        output_dir=args.output,
        so8t_config_path=args.so8t_config
    )
    
    result = replacer.replace()
    
    logger.info("="*80)
    logger.info("[COMPLETE] Model replacement completed!")
    logger.info(f"Result: {result}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

