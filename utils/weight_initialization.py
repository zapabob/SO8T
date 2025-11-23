#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Weight Initialization Utilities
重みの適切な初期化による崩壊防止
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class SO8TWeightInitializer:
    """
    SO8T専用の重み初期化クラス
    SO(8)群の特性を考慮した初期化
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        logger.info("SO8TWeightInitializer initialized")
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'rotation_init_scale': 0.1,
            'triality_init_scale': 0.05,
            'safety_init_scale': 0.02,
            'bias_init_value': 0.0,
            'use_xavier_uniform': True,
            'use_orthogonal_init': True,
            'preserve_so8_structure': True,
            'temperature_scaling': 1.0
        }
    
    def initialize_model(self, model: nn.Module) -> nn.Module:
        """
        モデル全体を初期化
        
        Args:
            model: 初期化対象のモデル
            
        Returns:
            初期化されたモデル
        """
        logger.info("Initializing SO8T model weights...")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._initialize_linear(module, name)
            elif isinstance(module, nn.Conv2d):
                self._initialize_conv2d(module, name)
            elif isinstance(module, nn.LSTM):
                self._initialize_lstm(module, name)
            elif isinstance(module, nn.GRU):
                self._initialize_gru(module, name)
            elif isinstance(module, nn.Transformer):
                self._initialize_transformer(module, name)
            elif isinstance(module, nn.MultiheadAttention):
                self._initialize_attention(module, name)
        
        # カスタムSO8Tモジュールの初期化
        self._initialize_so8_modules(model)
        
        logger.info("Model initialization completed")
        return model
    
    def _initialize_linear(self, module: nn.Linear, name: str):
        """Linear層の初期化"""
        if 'rotation' in name.lower():
            self._initialize_rotation_linear(module)
        elif 'triality' in name.lower() or 'task_head' in name.lower() or 'safety_head' in name.lower() or 'authority_head' in name.lower():
            self._initialize_triality_linear(module)
        elif 'safety' in name.lower():
            self._initialize_safety_linear(module)
        else:
            self._initialize_standard_linear(module)
    
    def _initialize_rotation_linear(self, module: nn.Linear):
        """回転重みの初期化"""
        # SO(8)群の特性を考慮した初期化
        if module.in_features == module.out_features and module.in_features == 8:
            # 8x8の回転行列として初期化
            self._initialize_rotation_matrix(module.weight)
        else:
            # 一般的な回転重みの初期化
            init.xavier_uniform_(module.weight, gain=self.config['rotation_init_scale'])
        
        if module.bias is not None:
            init.constant_(module.bias, self.config['bias_init_value'])
    
    def _initialize_triality_linear(self, module: nn.Linear):
        """三性重みの初期化"""
        # 三性推論の特性を考慮した初期化
        init.xavier_uniform_(module.weight, gain=self.config['triality_init_scale'])
        
        if module.bias is not None:
            # 三性のバランスを考慮したバイアス初期化
            bias_init = torch.randn(module.bias.size()) * self.config['triality_init_scale'] * 0.1
            module.bias.data = bias_init
    
    def _initialize_safety_linear(self, module: nn.Linear):
        """安全性重みの初期化"""
        # 安全性を重視した初期化
        init.xavier_uniform_(module.weight, gain=self.config['safety_init_scale'])
        
        if module.bias is not None:
            # 安全性に偏ったバイアス初期化
            safety_bias = torch.ones(module.bias.size()) * self.config['safety_init_scale'] * 0.5
            module.bias.data = safety_bias
    
    def _initialize_standard_linear(self, module: nn.Linear):
        """標準的なLinear層の初期化"""
        if self.config['use_xavier_uniform']:
            init.xavier_uniform_(module.weight)
        else:
            init.kaiming_uniform_(module.weight, nonlinearity='relu')
        
        if module.bias is not None:
            init.constant_(module.bias, self.config['bias_init_value'])
    
    def _initialize_conv2d(self, module: nn.Conv2d, name: str):
        """Conv2D層の初期化"""
        if 'rotation' in name.lower():
            init.xavier_uniform_(module.weight, gain=self.config['rotation_init_scale'])
        elif 'safety' in name.lower():
            init.xavier_uniform_(module.weight, gain=self.config['safety_init_scale'])
        else:
            init.kaiming_uniform_(module.weight, nonlinearity='relu')
        
        if module.bias is not None:
            init.constant_(module.bias, self.config['bias_init_value'])
    
    def _initialize_lstm(self, module: nn.LSTM, name: str):
        """LSTM層の初期化"""
        for name_param, param in module.named_parameters():
            if 'weight_ih' in name_param:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name_param:
                init.orthogonal_(param)
            elif 'bias' in name_param:
                init.constant_(param, 0)
                # 忘却ゲートのバイアスを1に設定
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def _initialize_gru(self, module: nn.GRU, name: str):
        """GRU層の初期化"""
        for name_param, param in module.named_parameters():
            if 'weight_ih' in name_param:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name_param:
                init.orthogonal_(param)
            elif 'bias' in name_param:
                init.constant_(param, 0)
    
    def _initialize_transformer(self, module: nn.Transformer, name: str):
        """Transformer層の初期化"""
        for name_param, param in module.named_parameters():
            if 'weight' in name_param:
                init.xavier_uniform_(param)
            elif 'bias' in name_param:
                init.constant_(param, 0)
    
    def _initialize_attention(self, module: nn.MultiheadAttention, name: str):
        """Attention層の初期化"""
        for name_param, param in module.named_parameters():
            if 'weight' in name_param:
                init.xavier_uniform_(param)
            elif 'bias' in name_param:
                init.constant_(param, 0)
    
    def _initialize_so8_modules(self, model: nn.Module):
        """SO8T特有のモジュールの初期化"""
        for name, module in model.named_modules():
            if hasattr(module, 'so8_rotation_params'):
                self._initialize_so8_rotation_params(module)
            elif hasattr(module, 'triality_heads'):
                self._initialize_triality_heads(module)
            elif hasattr(module, 'safety_judgment'):
                self._initialize_safety_judgment(module)
    
    def _initialize_so8_rotation_params(self, module):
        """SO8回転パラメータの初期化"""
        if hasattr(module, 'so8_rotation_params'):
            # SO(8)群の生成子を初期化
            init.xavier_uniform_(module.so8_rotation_params, gain=self.config['rotation_init_scale'])
    
    def _initialize_triality_heads(self, module):
        """三性ヘッドの初期化"""
        if hasattr(module, 'triality_heads'):
            for head in module.triality_heads:
                if hasattr(head, 'weight'):
                    init.xavier_uniform_(head.weight, gain=self.config['triality_init_scale'])
                if hasattr(head, 'bias') and head.bias is not None:
                    init.constant_(head.bias, 0)
    
    def _initialize_safety_judgment(self, module):
        """安全性判定の初期化"""
        if hasattr(module, 'safety_judgment'):
            for param in module.safety_judgment.parameters():
                if param.dim() > 1:
                    init.xavier_uniform_(param, gain=self.config['safety_init_scale'])
                else:
                    init.constant_(param, 0)
    
    def _initialize_rotation_matrix(self, weight: torch.Tensor):
        """回転行列の初期化"""
        # SO(8)群の要素として初期化
        n = weight.size(0)
        
        # ランダムな直交行列を生成
        if n == 8:
            # SO(8)群の生成子を使用
            generators = self._get_so8_generators()
            rotation_matrix = torch.eye(8)
            
            for i, generator in enumerate(generators):
                angle = torch.randn(1) * self.config['rotation_init_scale']
                rotation_matrix = rotation_matrix @ torch.matrix_exp(angle * generator)
            
            weight.data = rotation_matrix
        else:
            # 一般的な直交行列の初期化
            init.orthogonal_(weight)
    
    def _get_so8_generators(self) -> List[torch.Tensor]:
        """SO(8)群の生成子を取得"""
        generators = []
        
        # SO(8)群の生成子（反対称行列）
        for i in range(8):
            for j in range(i+1, 8):
                generator = torch.zeros(8, 8)
                generator[i, j] = 1
                generator[j, i] = -1
                generators.append(generator)
        
        return generators
    
    def initialize_with_pretrained(self, 
                                 model: nn.Module, 
                                 pretrained_path: str,
                                 strict: bool = True) -> nn.Module:
        """
        事前学習済み重みで初期化
        
        Args:
            model: 初期化対象のモデル
            pretrained_path: 事前学習済み重みのパス
            strict: 厳密なロードを行うか
            
        Returns:
            初期化されたモデル
        """
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 重みをロード
            model.load_state_dict(state_dict, strict=strict)
            
            logger.info("Pretrained weights loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Falling back to random initialization")
            model = self.initialize_model(model)
        
        return model
    
    def get_initialization_report(self, model: nn.Module) -> Dict[str, Any]:
        """初期化レポートを生成"""
        report = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'initialization_stats': {},
            'layer_analysis': {}
        }
        
        for name, param in model.named_parameters():
            report['total_parameters'] += param.numel()
            if param.requires_grad:
                report['trainable_parameters'] += param.numel()
            
            # 重みの統計
            weight_data = param.data.cpu().numpy()
            stats = {
                'mean': float(np.mean(weight_data)),
                'std': float(np.std(weight_data)),
                'min': float(np.min(weight_data)),
                'max': float(np.max(weight_data)),
                'norm': float(np.linalg.norm(weight_data))
            }
            
            report['layer_analysis'][name] = stats
            
            # カテゴリ別の統計
            if 'rotation' in name.lower():
                category = 'rotation'
            elif 'triality' in name.lower() or 'task_head' in name.lower() or 'safety_head' in name.lower() or 'authority_head' in name.lower():
                category = 'triality'
            elif 'safety' in name.lower():
                category = 'safety'
            else:
                category = 'standard'
            
            if category not in report['initialization_stats']:
                report['initialization_stats'][category] = []
            
            report['initialization_stats'][category].append(stats)
        
        # カテゴリ別の平均統計
        for category, stats_list in report['initialization_stats'].items():
            if stats_list:
                report['initialization_stats'][category] = {
                    'mean_mean': np.mean([s['mean'] for s in stats_list]),
                    'mean_std': np.mean([s['std'] for s in stats_list]),
                    'mean_norm': np.mean([s['norm'] for s in stats_list]),
                    'layer_count': len(stats_list)
                }
        
        return report

def initialize_so8t_model(model: nn.Module, 
                         config: Optional[Dict] = None,
                         pretrained_path: Optional[str] = None) -> nn.Module:
    """
    SO8Tモデルを初期化
    
    Args:
        model: 初期化対象のモデル
        config: 設定パラメータ
        pretrained_path: 事前学習済み重みのパス
        
    Returns:
        初期化されたモデル
    """
    initializer = SO8TWeightInitializer(config)
    
    if pretrained_path and os.path.exists(pretrained_path):
        return initializer.initialize_with_pretrained(model, pretrained_path)
    else:
        return initializer.initialize_model(model)

if __name__ == "__main__":
    # テスト用のダミーモデル
    class DummySO8TModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotation_layer = nn.Linear(8, 8)
            self.task_head = nn.Linear(10, 3)
            self.safety_head = nn.Linear(10, 3)
            self.authority_head = nn.Linear(10, 3)
            self.safety_judgment = nn.Linear(5, 1)
    
    model = DummySO8TModel()
    initializer = SO8TWeightInitializer()
    
    # 初期化実行
    initialized_model = initializer.initialize_model(model)
    
    # レポート生成
    report = initializer.get_initialization_report(initialized_model)
    
    print("SO8T Weight Initialization Test")
    print("=" * 50)
    print(f"Total parameters: {report['total_parameters']:,}")
    print(f"Trainable parameters: {report['trainable_parameters']:,}")
    print("\nInitialization stats by category:")
    for category, stats in report['initialization_stats'].items():
        print(f"  {category}: {stats}")





