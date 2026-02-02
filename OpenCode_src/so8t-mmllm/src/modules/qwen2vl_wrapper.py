"""
Qwen2VLForConditionalGeneration用SO8T回転ゲートラッパ
SDPA出力後に回転ゲートを挿入する非破壊的拡張
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import Qwen2VLForConditionalGeneration
from .rotation_gate import SO8TRotationGate


class SO8TQwen2VLWrapper(nn.Module):
    """
    Qwen2VLForConditionalGeneration用SO8T回転ゲートラッパ
    既存モデルを非破壊的に拡張
    """
    
    def __init__(
        self,
        base_model: Qwen2VLForConditionalGeneration,
        rotation_enabled: bool = True,
        rotation_layers: Optional[List[int]] = None,
        init_scale: float = 0.1
    ):
        """
        Args:
            base_model: ベースとなるQwen2VLForConditionalGenerationモデル
            rotation_enabled: 回転ゲートを有効にするか
            rotation_layers: 回転ゲートを適用するレイヤー番号（Noneの場合は全レイヤー）
            init_scale: 回転パラメータの初期化スケール
        """
        super().__init__()
        
        self.base_model = base_model
        self.rotation_enabled = rotation_enabled
        self.rotation_layers = rotation_layers
        self.init_scale = init_scale
        
        # モデル設定を取得
        self.hidden_size = base_model.config.hidden_size
        self.num_layers = base_model.config.num_hidden_layers
        
        # 回転ゲートを初期化
        if rotation_enabled:
            self._setup_rotation_gates()
        
        # 元のforwardメソッドを保存
        self._original_forward = base_model.forward
        self._original_generate = base_model.generate
        
        # ラッパされたforwardメソッドを設定
        base_model.forward = self._wrapped_forward
        base_model.generate = self._wrapped_generate
    
    def _setup_rotation_gates(self) -> None:
        """回転ゲートをセットアップ"""
        self.rotation_gates = nn.ModuleDict()
        
        # 適用するレイヤーを決定
        if self.rotation_layers is None:
            # 全レイヤーに適用
            target_layers = list(range(self.num_layers))
        else:
            target_layers = self.rotation_layers
        
        # 各レイヤーに回転ゲートを追加
        for layer_idx in target_layers:
            if hasattr(self.base_model.model, 'layers') and layer_idx < len(self.base_model.model.layers):
                gate_name = f"layer_{layer_idx}"
                self.rotation_gates[gate_name] = SO8TRotationGate(
                    hidden_size=self.hidden_size,
                    init_scale=self.init_scale,
                    learnable=True
                )
    
    def _apply_rotation_gate(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        指定されたレイヤーで回転ゲートを適用
        
        Args:
            hidden_states: 隠れ状態 [B, T, D]
            layer_idx: レイヤー番号
            
        Returns:
            回転適用後の隠れ状態
        """
        if not self.rotation_enabled:
            return hidden_states
        
        gate_name = f"layer_{layer_idx}"
        if gate_name in self.rotation_gates:
            return self.rotation_gates[gate_name](hidden_states)
        
        return hidden_states
    
    def _wrapped_forward(self, *args, **kwargs):
        """ラッパされたforwardメソッド"""
        # 元のforwardを呼び出し
        outputs = self._original_forward(*args, **kwargs)
        
        # 回転ゲートを適用（hidden_statesがある場合）
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            rotated_hidden_states = []
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                rotated_state = self._apply_rotation_gate(hidden_state, layer_idx)
                rotated_hidden_states.append(rotated_state)
            outputs.hidden_states = tuple(rotated_hidden_states)
        
        return outputs
    
    def _wrapped_generate(self, *args, **kwargs):
        """ラッパされたgenerateメソッド"""
        # 元のgenerateを呼び出し
        return self._original_generate(*args, **kwargs)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """前向き計算（回転ゲート適用）"""
        return self.base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """テキスト生成（回転ゲート適用）"""
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def get_rotation_matrices(self) -> dict:
        """回転行列を取得"""
        if not self.rotation_enabled:
            return {}
        
        matrices = {}
        for name, gate in self.rotation_gates.items():
            matrices[name] = gate.get_rotation_matrices()
        
        return matrices
    
    def enable_rotation(self, layer_indices: Optional[List[int]] = None) -> None:
        """回転ゲートを有効化"""
        self.rotation_enabled = True
        if layer_indices is not None:
            self.rotation_layers = layer_indices
        self._setup_rotation_gates()
    
    def disable_rotation(self) -> None:
        """回転ゲートを無効化"""
        self.rotation_enabled = False
        if hasattr(self, 'rotation_gates'):
            self.rotation_gates.clear()
    
    def save_rotation_parameters(self, filepath: str) -> None:
        """回転パラメータを保存"""
        if self.rotation_enabled and hasattr(self, 'rotation_gates'):
            torch.save({
                name: gate.theta for name, gate in self.rotation_gates.items()
            }, filepath)
    
    def load_rotation_parameters(self, filepath: str) -> None:
        """回転パラメータを読み込み"""
        if self.rotation_enabled and hasattr(self, 'rotation_gates'):
            params = torch.load(filepath, map_location=self.base_model.device)
            for name, theta in params.items():
                if name in self.rotation_gates:
                    self.rotation_gates[name].theta.data = theta


class SO8TQwen2VLAttentionWrapper(nn.Module):
    """
    Qwen2VLの個別アテンション層用ラッパ
    より細かい制御が必要な場合に使用
    """
    
    def __init__(
        self,
        attention_layer: nn.Module,
        hidden_size: int,
        rotation_enabled: bool = True,
        init_scale: float = 0.1
    ):
        """
        Args:
            attention_layer: 元のアテンション層
            hidden_size: 隠れ層サイズ
            rotation_enabled: 回転ゲートを有効にするか
            init_scale: 初期化スケール
        """
        super().__init__()
        
        self.attention_layer = attention_layer
        self.hidden_size = hidden_size
        self.rotation_enabled = rotation_enabled
        
        # 回転ゲートを初期化
        if rotation_enabled:
            self.rotation_gate = SO8TRotationGate(
                hidden_size=hidden_size,
                init_scale=init_scale,
                learnable=True
            )
        else:
            self.rotation_gate = None
        
        # 元のforwardメソッドを保存
        self._original_forward = attention_layer.forward
    
        # ラッパされたforwardメソッドを設定
        attention_layer.forward = self._wrapped_forward
    
    def _wrapped_forward(self, *args, **kwargs):
        """ラッパされたforwardメソッド"""
        # 元のアテンション計算を実行
        outputs = self._original_forward(*args, **kwargs)
        
        # 回転ゲートを適用
        if self.rotation_gate is not None and hasattr(outputs, 'last_hidden_state'):
            outputs.last_hidden_state = self.rotation_gate(outputs.last_hidden_state)
        elif self.rotation_gate is not None and isinstance(outputs, tuple):
            # タプルの場合、最初の要素が隠れ状態と仮定
            if len(outputs) > 0:
                rotated_state = self.rotation_gate(outputs[0])
                outputs = (rotated_state,) + outputs[1:]
        
        return outputs
    
    def get_rotation_matrix(self) -> Optional[torch.Tensor]:
        """回転行列を取得"""
        if self.rotation_gate is not None:
            return self.rotation_gate.get_rotation_matrices()
        return None


def create_so8t_qwen2vl_model(
    model_path: str,
    rotation_enabled: bool = True,
    rotation_layers: Optional[List[int]] = None,
    init_scale: float = 0.1,
    device_map: str = "auto"
) -> SO8TQwen2VLWrapper:
    """
    SO8T回転ゲート付きQwen2VLモデルを作成
    
    Args:
        model_path: モデルパス
        rotation_enabled: 回転ゲートを有効にするか
        rotation_layers: 回転ゲートを適用するレイヤー
        init_scale: 初期化スケール
        device_map: デバイスマップ
        
    Returns:
        SO8TQwen2VLWrapperインスタンス
    """
    # ベースモデルを読み込み
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )
    
    # ラッパを作成
    wrapper = SO8TQwen2VLWrapper(
        base_model=base_model,
        rotation_enabled=rotation_enabled,
        rotation_layers=rotation_layers,
        init_scale=init_scale
    )
    
    return wrapper
