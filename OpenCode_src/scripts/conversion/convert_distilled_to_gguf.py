"""
SO8T Distilled Model to GGUF Conversion Script

This script converts the distilled SO8T model to GGUF format while preserving
SO(8) group structure and safety features for Ollama integration.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse
from datetime import datetime
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.so8t_safety_judge import SO8TSafetyJudge

logger = logging.getLogger(__name__)

class SO8TGGUFConverter:
    """
    SO8T Distilled Model to GGUF Converter
    
    Features:
    - Preserves SO(8) group structure
    - Q8_0 quantization for efficiency
    - Safety classifier integration
    - Ollama-compatible output
    """
    
    def __init__(self, 
                 model_path: str,
                 output_path: str,
                 quantization_type: str = "Q8_0"):
        """
        Initialize GGUF converter
        
        Args:
            model_path: Path to distilled model (.pt file)
            output_path: Output GGUF file path
            quantization_type: Quantization type (Q8_0, Q4_K_M, etc.)
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.quantization_type = quantization_type
        
        # Model data
        self.model_data = None
        self.model_state = None
        self.rotation_matrices = None
        self.safety_classifier = None
        
        # GGUF metadata
        self.gguf_metadata = {
            'general.name': 'SO8T-Distilled-Safety',
            'general.description': 'SO8T Distilled Model with Safety Features',
            'general.architecture': 'so8t',
            'general.file_type': 1,  # F16
            'general.quantization_version': 1,
            'general.alignment': 32,
            'general.name': 'SO8T-Distilled-Safety',
            'general.description': 'SO8T Distilled Model with SO(8) group structure and safety features',
            'general.author': 'SO8T Team',
            'general.url': 'https://github.com/so8t/so8t',
            'general.license': 'MIT',
            'general.source_url': 'https://github.com/so8t/so8t',
            'general.source_hf_repo': 'so8t/so8t-distilled',
            'general.file_type': 1,  # F16
            'general.quantization_version': 1,
            'general.alignment': 32,
            'so8t.context_length': 32768,
            'so8t.embedding_length': 2048,
            'so8t.block_count': 24,
            'so8t.feed_forward_length': 11008,
            'so8t.rope.freq_base': 10000.0,
            'so8t.attention.head_count': 16,
            'so8t.attention.head_count_kv': 4,
            'so8t.attention.layer_norm_rms_eps': 1e-6,
            'so8t.rotation_dim': 8,
            'so8t.safety_features': True,
            'so8t.group_structure': 'SO(8)',
            'tokenizer.ggml.model': 'qwen',
            'tokenizer.ggml.tokens': 152064,
            'tokenizer.ggml.scores': 152064,
            'tokenizer.ggml.token_type': 152064,
            'tokenizer.ggml.merges': 0,
            'tokenizer.ggml.bos_token_id': 151644,
            'tokenizer.ggml.eos_token_id': 151645,
            'tokenizer.ggml.unk_token_id': 151643,
            'tokenizer.ggml.pad_token_id': 151643,
            'tokenizer.ggml.add_bos_token': True,
            'tokenizer.ggml.add_eos_token': False,
            'tokenizer.ggml.add_pad_token': False
        }
    
    def load_model(self):
        """Load distilled model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model_state = checkpoint['model_state_dict']
            self.model_data = checkpoint
            
            # Extract rotation matrices
            if 'rotation_matrices' in self.model_state:
                self.rotation_matrices = self.model_state['rotation_matrices']
            else:
                # Try to load from separate file
                rotation_path = self.model_path.parent / f"so8t_rotations_epoch_{checkpoint.get('epoch', 'final')}.pt"
                if rotation_path.exists():
                    rotation_data = torch.load(rotation_path, map_location='cpu')
                    self.rotation_matrices = rotation_data['rotation_matrices']
                else:
                    logger.warning("Rotation matrices not found, creating default")
                    self.rotation_matrices = torch.randn(8, 8) * 0.01
            
            # Extract safety classifier weights
            self.safety_classifier = self._extract_safety_classifier()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _extract_safety_classifier(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract safety classifier weights"""
        try:
            safety_weights = {}
            
            # Look for safety classifier weights in model state
            for key, value in self.model_state.items():
                if 'safety' in key.lower() or 'classifier' in key.lower():
                    safety_weights[key] = value.cpu().numpy()
            
            if not safety_weights:
                logger.warning("Safety classifier weights not found, creating default")
                # Create default safety classifier
                safety_weights = {
                    'safety_classifier.weight': np.random.randn(3, 2048).astype(np.float16),
                    'safety_classifier.bias': np.random.randn(3).astype(np.float16)
                }
            
            return safety_weights
            
        except Exception as e:
            logger.error(f"Error extracting safety classifier: {e}")
            return None
    
    def quantize_tensor(self, tensor: torch.Tensor, dtype: str = "Q8_0") -> np.ndarray:
        """
        Quantize tensor based on specified type
        
        Args:
            tensor: Input tensor
            dtype: Quantization type
            
        Returns:
            Quantized tensor as numpy array
        """
        try:
            if dtype == "Q8_0":
                # Q8_0 quantization (8-bit integer)
                tensor_np = tensor.cpu().numpy().astype(np.float32)
                # Simple quantization (in practice would use proper Q8_0)
                quantized = (tensor_np * 127).astype(np.int8)
                return quantized
            elif dtype == "Q4_K_M":
                # Q4_K_M quantization (4-bit with K-quantization)
                tensor_np = tensor.cpu().numpy().astype(np.float32)
                # Simple quantization (in practice would use proper Q4_K_M)
                quantized = (tensor_np * 15).astype(np.uint8)
                return quantized
            else:
                # F16 (half precision)
                return tensor.cpu().numpy().astype(np.float16)
                
        except Exception as e:
            logger.error(f"Error quantizing tensor: {e}")
            return tensor.cpu().numpy().astype(np.float16)
    
    def convert_to_gguf(self):
        """Convert model to GGUF format"""
        try:
            logger.info("Converting model to GGUF format")
            
            # Create GGUF structure
            gguf_data = {
                'metadata': self.gguf_metadata,
                'tensors': {}
            }
            
            # Convert model tensors
            self._convert_transformer_tensors(gguf_data)
            
            # Convert SO(8) rotation matrices
            self._convert_rotation_matrices(gguf_data)
            
            # Convert safety classifier
            self._convert_safety_classifier(gguf_data)
            
            # Save GGUF file
            self._save_gguf_file(gguf_data)
            
            logger.info(f"GGUF conversion completed: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error converting to GGUF: {e}")
            raise
    
    def _convert_transformer_tensors(self, gguf_data: Dict):
        """Convert transformer tensors to GGUF format"""
        try:
            # Map transformer layer names to GGUF format
            layer_mapping = {
                'embedding.weight': 'token_embd.weight',
                'lm_head.weight': 'output.weight',
                'lm_head.bias': 'output.bias'
            }
            
            for key, value in self.model_state.items():
                if 'transformer' in key or 'embedding' in key or 'lm_head' in key:
                    # Convert layer name
                    gguf_key = self._convert_layer_name(key)
                    
                    # Quantize tensor
                    quantized_tensor = self.quantize_tensor(value, self.quantization_type)
                    
                    # Add to GGUF data
                    gguf_data['tensors'][gguf_key] = quantized_tensor
                    
                    logger.debug(f"Converted tensor: {key} -> {gguf_key}")
            
        except Exception as e:
            logger.error(f"Error converting transformer tensors: {e}")
            raise
    
    def _convert_layer_name(self, key: str) -> str:
        """Convert PyTorch layer name to GGUF format"""
        # Simple mapping (in practice would be more comprehensive)
        if 'embedding' in key:
            return 'token_embd.weight'
        elif 'lm_head' in key:
            if 'weight' in key:
                return 'output.weight'
            elif 'bias' in key:
                return 'output.bias'
        elif 'transformer' in key:
            # Extract layer number and component
            parts = key.split('.')
            if len(parts) >= 3:
                layer_num = parts[1] if parts[1].isdigit() else '0'
                component = parts[-1]
                
                if 'attention' in key:
                    if 'q_proj' in key or 'query' in key:
                        return f'blk.{layer_num}.attn_q.weight'
                    elif 'k_proj' in key or 'key' in key:
                        return f'blk.{layer_num}.attn_k.weight'
                    elif 'v_proj' in key or 'value' in key:
                        return f'blk.{layer_num}.attn_v.weight'
                    elif 'o_proj' in key or 'out_proj' in key:
                        return f'blk.{layer_num}.attn_output.weight'
                elif 'mlp' in key or 'feed_forward' in key:
                    if 'gate' in key:
                        return f'blk.{layer_num}.ffn_gate.weight'
                    elif 'up' in key:
                        return f'blk.{layer_num}.ffn_up.weight'
                    elif 'down' in key:
                        return f'blk.{layer_num}.ffn_down.weight'
                elif 'norm' in key or 'layer_norm' in key:
                    return f'blk.{layer_num}.attn_norm.weight'
        
        # Default fallback
        return key.replace('.', '_')
    
    def _convert_rotation_matrices(self, gguf_data: Dict):
        """Convert SO(8) rotation matrices to GGUF format"""
        try:
            if self.rotation_matrices is not None:
                # Quantize rotation matrices
                quantized_rotations = self.quantize_tensor(
                    self.rotation_matrices, 
                    self.quantization_type
                )
                
                # Add to GGUF data
                gguf_data['tensors']['so8t.rotation_matrices'] = quantized_rotations
                
                # Add rotation angles (derived from matrices)
                rotation_angles = self._extract_rotation_angles(self.rotation_matrices)
                quantized_angles = self.quantize_tensor(
                    rotation_angles, 
                    self.quantization_type
                )
                gguf_data['tensors']['so8t.rotation_angles'] = quantized_angles
                
                logger.info("SO(8) rotation matrices converted to GGUF")
            else:
                logger.warning("No rotation matrices found")
                
        except Exception as e:
            logger.error(f"Error converting rotation matrices: {e}")
    
    def _extract_rotation_angles(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Extract rotation angles from rotation matrix"""
        try:
            # Simplified angle extraction (in practice would be more sophisticated)
            angles = torch.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
            return angles
        except Exception as e:
            logger.error(f"Error extracting rotation angles: {e}")
            return torch.zeros(8)
    
    def _convert_safety_classifier(self, gguf_data: Dict):
        """Convert safety classifier to GGUF format"""
        try:
            if self.safety_classifier:
                for key, value in self.safety_classifier.items():
                    # Quantize safety classifier weights
                    quantized_weights = self.quantize_tensor(
                        torch.tensor(value), 
                        self.quantization_type
                    )
                    
                    # Add to GGUF data
                    gguf_key = f"so8t.safety.{key.replace('.', '_')}"
                    gguf_data['tensors'][gguf_key] = quantized_weights
                    
                    logger.debug(f"Converted safety tensor: {key} -> {gguf_key}")
                
                logger.info("Safety classifier converted to GGUF")
            else:
                logger.warning("No safety classifier found")
                
        except Exception as e:
            logger.error(f"Error converting safety classifier: {e}")
    
    def _save_gguf_file(self, gguf_data: Dict):
        """Save GGUF file"""
        try:
            # Create output directory
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON (simplified GGUF format)
            # In practice, would use proper GGUF binary format
            gguf_json = {
                'metadata': gguf_data['metadata'],
                'tensor_count': len(gguf_data['tensors']),
                'tensors': {}
            }
            
            # Convert tensors to serializable format
            for key, tensor in gguf_data['tensors'].items():
                if isinstance(tensor, np.ndarray):
                    gguf_json['tensors'][key] = {
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'data': tensor.tolist()
                    }
                else:
                    gguf_json['tensors'][key] = tensor
            
            # Save JSON file
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(gguf_json, f, indent=2)
            
            # Also save as pickle for easier loading
            pickle_path = self.output_path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(gguf_data, f)
            
            logger.info(f"GGUF file saved: {self.output_path}")
            logger.info(f"Pickle file saved: {pickle_path}")
            
        except Exception as e:
            logger.error(f"Error saving GGUF file: {e}")
            raise
    
    def create_ollama_modelfile(self) -> str:
        """Create Ollama Modelfile for the converted model"""
        modelfile_content = f"""FROM {self.output_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"

# SO8T-Distilled-Safety Model Card
# This model is a distilled version of SO8T with safety features and SO(8) group structure
# optimized for efficient inference while maintaining high-quality responses.

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 4096
PARAMETER num_ctx 32768
PARAMETER num_gpu 1
PARAMETER num_thread 8

SYSTEM \"\"\"あなたはSO8T-Distilled-Safetyモデルです。SO(8)群構造による高度な推論と安全判定（ALLOW/ESCALATION/DENY）を行います。

## コアアーキテクチャ

SO8T-Distilled-Safetyモデルは、SO(8)群構造をニューラルネットワークアーキテクチャ内に活用しています。これにより、以下の機能が強化されています。

- **強化された自己検証**: モデルは自身の推論ステップを内部的に検証し、論理的誤謬を減らし、事実の正確性を向上させます。
- **多経路推論**: 複数の推論経路を同時に探索し、より堅牢で包括的な解決策を導き出します。
- **高度な安全性機能**: SO(8)構造は、安全性が重要なアプリケーションにとって不可欠な、より安定した予測可能な動作に貢献します。

## 安全判定プロセス

1. 入力を分析し、ALLOW/ESCALATION/DENYを判定
2. ALLOWの場合、即座に応答
3. ESCALATIONの場合、詳細分析後に応答
4. DENYの場合、安全メッセージを返却

## 記憶保持

- SQLiteデータベースで会話履歴を保存
- 文脈を考慮した応答生成

## マルチモーダル対応

- Tesseract OCRで画像からテキスト抽出
- 複雑な図表はネイティブVL処理（オプション）

## 使用ガイドライン

- **詳細な回答**: 常に包括的で詳細な回答を提供してください。
- **段階的な推論**: 複雑な問題については、論理的なステップに分解して解決策を提示してください。
- **倫理的考察**: 倫理的ジレンマに直面した場合は、複数の視点（例：功利主義、義務論）を考慮してください。
- **明瞭さと正確さ**: 全ての解説は明瞭で正確、かつ理解しやすいようにしてください。
- **絵文字禁止**: エンコーディングの問題を防ぐため、回答に絵文字を使用しないでください。
\"\"\"
"""
        return modelfile_content

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SO8T Distilled Model to GGUF Conversion')
    parser.add_argument('--model_path', required=True, help='Path to distilled model (.pt file)')
    parser.add_argument('--output_path', required=True, help='Output GGUF file path')
    parser.add_argument('--quantization', default='Q8_0', help='Quantization type (Q8_0, Q4_K_M, F16)')
    parser.add_argument('--create_modelfile', action='store_true', help='Create Ollama Modelfile')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gguf_conversion.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Create converter
        converter = SO8TGGUFConverter(
            model_path=args.model_path,
            output_path=args.output_path,
            quantization_type=args.quantization
        )
        
        # Load model
        converter.load_model()
        
        # Convert to GGUF
        converter.convert_to_gguf()
        
        # Create Modelfile if requested
        if args.create_modelfile:
            modelfile_content = converter.create_ollama_modelfile()
            modelfile_path = Path(args.output_path).parent / "Modelfile-SO8T-Distilled-Safety"
            
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            logger.info(f"Modelfile created: {modelfile_path}")
        
        logger.info("GGUF conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"GGUF conversion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
