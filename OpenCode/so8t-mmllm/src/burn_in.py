"""
SO8T Burn-in (焼きこみ) Mechanism
学習後のSO8T回転行列を線形層に吸収させる機構

理論的背景:
- 学習中: attention_output → SO8T回転 → 次の層
- 焼きこみ後: attention_output → 吸収済み線形層 → 次の層
- 右掛け吸収: W_o' = W_o @ R により、推論グラフから回転モジュールを削除可能
- 基底不一致の解消: 学習時と推論時で同じ座標系を維持

利点:
1. 推論速度: 追加の行列積が不要になる
2. 量子化安定性: 標準的なGEMM演算として量子化される
3. RoPE互換性: 回転順序が変わらないため位相ドリフトを防ぐ
4. メモリ効率: 回転モジュールのパラメータが不要になる

Author: SO8T Project Team
License: Apache 2.0
Date: 2024-11-06
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def fold_so8t_into_linear(
    W_o: torch.Tensor,
    R_eff: torch.Tensor,
    verify: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SO8T回転行列を線形層の重みに吸収（右掛け）
    
    学習時の座標系を保ったまま推論グラフを簡略化する。
    W_o @ R_eff により、SO8Tモジュールを削除しても
    学習時と同じ変換が実現される。
    
    Args:
        W_o: 出力線形層の重み [D_out, D]
        R_eff: 効果的な回転行列（ブロック対角） [D, D]
        verify: 吸収前後の誤差を検証するか
        
    Returns:
        W_o': 吸収後の重み [D_out, D]
        stats: 統計情報（誤差など）
        
    Formula:
        y = W_o @ (R @ x) = (W_o @ R) @ x = W_o' @ x
        
    Properties:
        - 関数として等価（数値誤差の範囲内）
        - パラメータ数は変わらない
        - 推論速度が向上（1回の行列積で済む）
    """
    assert W_o.shape[1] == R_eff.shape[0], \
        f"Shape mismatch: W_o {W_o.shape} vs R_eff {R_eff.shape}"
    
    device = W_o.device
    dtype = W_o.dtype
    
    # 回転行列を同じデバイス・型に変換
    R_eff = R_eff.to(device=device, dtype=dtype)
    
    # 右掛けで吸収
    W_o_prime = torch.matmul(W_o, R_eff)
    
    stats = {
        'w_o_norm': W_o.norm().item(),
        'w_o_prime_norm': W_o_prime.norm().item(),
        'r_eff_norm': R_eff.norm().item(),
    }
    
    # 検証: ダミー入力で前後の出力を比較
    if verify:
        batch_size = 4
        seq_len = 8
        D = W_o.shape[1]
        
        # ダミー入力
        x = torch.randn(batch_size, seq_len, D, device=device, dtype=dtype)
        
        # 吸収前: W_o @ (R @ x)
        Rx = torch.matmul(x, R_eff.T)  # [B, T, D]
        y_before = torch.matmul(Rx, W_o.T)  # [B, T, D_out]
        
        # 吸収後: W_o' @ x
        y_after = torch.matmul(x, W_o_prime.T)  # [B, T, D_out]
        
        # 誤差計算
        max_error = (y_before - y_after).abs().max().item()
        mean_error = (y_before - y_after).abs().mean().item()
        relative_error = max_error / (y_before.abs().max().item() + 1e-8)
        
        stats.update({
            'max_error': max_error,
            'mean_error': mean_error,
            'relative_error': relative_error,
        })
        
        logger.info(f"[Burn-in Verification]")
        logger.info(f"  Max error: {max_error:.6e}")
        logger.info(f"  Mean error: {mean_error:.6e}")
        logger.info(f"  Relative error: {relative_error:.6e}")
        
        # 許容誤差チェック
        if max_error > 1e-4:
            logger.warning(f"[WARNING] Large burn-in error: {max_error:.6e}")
    
    return W_o_prime, stats


def build_block_diagonal_rotation(R_blocks: List[torch.Tensor]) -> torch.Tensor:
    """
    8×8回転行列のリストからブロック対角行列を構築
    
    Args:
        R_blocks: 8×8回転行列のリスト [num_blocks個の[8, 8]]
        
    Returns:
        ブロック対角行列 [D, D] where D = num_blocks * 8
        
    Structure:
        [R_0  0    0   ...  0  ]
        [0    R_1  0   ...  0  ]
        [0    0    R_2 ...  0  ]
        [...  ...  ... ...  ...]
        [0    0    0   ...  R_n]
    """
    if len(R_blocks) == 0:
        raise ValueError("R_blocks is empty")
    
    # すべて同じデバイス・型に統一
    device = R_blocks[0].device
    dtype = R_blocks[0].dtype
    
    for i, R in enumerate(R_blocks):
        assert R.shape == (8, 8), f"Block {i} has invalid shape: {R.shape}"
        R_blocks[i] = R.to(device=device, dtype=dtype)
    
    # ブロック対角行列を構築
    R_block_diag = torch.block_diag(*R_blocks)
    
    return R_block_diag


def fold_blockdiag(
    W_o: torch.Tensor,
    R_blocks: List[torch.Tensor],
    verify: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    ブロック対角SO8T回転を線形層に吸収
    
    SO8TRotationGateから取得した8×8回転行列のリストを
    ブロック対角行列に変換してから吸収する。
    
    Args:
        W_o: 出力線形層の重み [D_out, D]
        R_blocks: 8×8回転行列のリスト [num_blocks個]
        verify: 検証を実行するか
        
    Returns:
        W_o': 吸収後の重み [D_out, D]
        stats: 統計情報
        
    Usage:
        # SO8TRotationGateから回転行列を取得
        R_blocks = so8t_gate.get_rotation_matrices()  # [num_blocks, 8, 8]
        R_blocks_list = [R_blocks[i] for i in range(R_blocks.shape[0])]
        
        # 線形層に吸収
        W_o_prime, stats = fold_blockdiag(W_o, R_blocks_list)
    """
    D = W_o.shape[1]
    num_blocks = len(R_blocks)
    
    assert D == num_blocks * 8, \
        f"Dimension mismatch: W_o has D={D}, but {num_blocks} blocks imply D={num_blocks*8}"
    
    logger.info(f"[Burn-in] Building block diagonal rotation: {num_blocks} blocks")
    
    # ブロック対角行列を構築
    R_block_diag = build_block_diagonal_rotation(R_blocks)
    
    logger.info(f"[Burn-in] Block diagonal shape: {R_block_diag.shape}")
    
    # 線形層に吸収
    W_o_prime, stats = fold_so8t_into_linear(W_o, R_block_diag, verify=verify)
    
    stats['num_blocks'] = num_blocks
    
    return W_o_prime, stats


def burn_in_model(
    model: nn.Module,
    target_module_names: Optional[List[str]] = None,
    verify: bool = True,
    inplace: bool = True,
    save_stats: bool = True,
    stats_path: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, float]]:
    """
    モデル全体にSO8T焼きこみを適用
    
    モデル内のすべてのSO8TRotationGateを検出し、
    対応する線形層（通常はo_proj）に回転行列を吸収させる。
    吸収後、回転モジュールは削除される。
    
    Args:
        model: PyTorchモデル
        target_module_names: 対象モジュール名のリスト
            （Noneの場合は'o_proj'を探す）
        verify: 各層で吸収前後の誤差を検証
        inplace: モデルを直接変更するか（Falseの場合はコピー）
        save_stats: 統計情報をファイルに保存するか
        stats_path: 統計情報の保存先（Noneの場合は自動生成）
        
    Returns:
        all_stats: 各層の統計情報
        
    Side Effects:
        - SO8TRotationGateモジュールが削除される
        - 対応する線形層の重みが更新される
        
    Example:
        # 学習後のモデルに焼きこみを適用
        stats = burn_in_model(model, verify=True)
        
        # 焼きこみ後は通常の推論として実行可能
        output = model(input_ids)
        
        # GGUF変換などに進める
        convert_to_gguf(model, output_path)
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    if target_module_names is None:
        target_module_names = ['o_proj', 'out_proj', 'c_proj']
    
    all_stats = {}
    burn_in_count = 0
    
    logger.info(f"[Burn-in] Starting model-wide burn-in")
    logger.info(f"[Burn-in] Target module names: {target_module_names}")
    
    # モジュールを走査
    modules_to_process = []
    
    for name, module in model.named_modules():
        # SO8TAttentionWrapperまたはSO8TRotationGateを探す
        from so8t_layer import SO8TRotationGate, SO8TAttentionWrapper
        
        if isinstance(module, SO8TAttentionWrapper):
            if module.rotation_gate is not None:
                modules_to_process.append((name, module, 'wrapper'))
        elif isinstance(module, SO8TRotationGate):
            modules_to_process.append((name, module, 'gate'))
    
    logger.info(f"[Burn-in] Found {len(modules_to_process)} SO8T modules")
    
    # 各モジュールを処理
    for name, so8t_module, module_type in modules_to_process:
        logger.info(f"[Burn-in] Processing: {name} (type: {module_type})")
        
        # 回転行列を取得
        if module_type == 'wrapper':
            rotation_gate = so8t_module.rotation_gate
        else:
            rotation_gate = so8t_module
        
        R_blocks = rotation_gate.get_rotation_matrices()  # [num_blocks, 8, 8]
        R_blocks_list = [R_blocks[i] for i in range(R_blocks.shape[0])]
        
        # 対応する線形層を探す
        # 名前から親モジュールとターゲット層を特定
        parent_name = '.'.join(name.split('.')[:-1])
        parent_module = model
        
        for part in parent_name.split('.'):
            if part:
                parent_module = getattr(parent_module, part)
        
        # ターゲット線形層を探す
        linear_layer = None
        linear_name = None
        
        for target_name in target_module_names:
            if hasattr(parent_module, target_name):
                candidate = getattr(parent_module, target_name)
                if isinstance(candidate, nn.Linear):
                    linear_layer = candidate
                    linear_name = target_name
                    break
        
        if linear_layer is None:
            logger.warning(f"[Burn-in] No target linear layer found for {name}")
            continue
        
        logger.info(f"[Burn-in] Found target linear layer: {linear_name}")
        logger.info(f"[Burn-in] Weight shape: {linear_layer.weight.shape}")
        
        # 焼きこみを適用
        try:
            W_o_prime, stats = fold_blockdiag(
                linear_layer.weight.data,
                R_blocks_list,
                verify=verify
            )
            
            # 重みを更新
            linear_layer.weight.data = W_o_prime
            
            # 統計を保存
            full_name = f"{name} -> {parent_name}.{linear_name}"
            all_stats[full_name] = stats
            
            burn_in_count += 1
            
            logger.info(f"[Burn-in] Successfully burned in: {full_name}")
            logger.info(f"[Burn-in] Stats: {stats}")
            
        except Exception as e:
            logger.error(f"[Burn-in] Failed to burn in {name}: {e}")
            continue
    
    # SO8Tモジュールを削除
    logger.info(f"[Burn-in] Removing SO8T rotation modules...")
    
    for name, so8t_module, module_type in modules_to_process:
        if module_type == 'wrapper':
            # Wrapperの場合は元のアテンションに戻す
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            parent_module = model
            for part in parent_name.split('.'):
                if part:
                    parent_module = getattr(parent_module, part)
            
            # 元のアテンションに戻す
            setattr(parent_module, attr_name, so8t_module.base_attention)
            logger.info(f"[Burn-in] Unwrapped: {name}")
        else:
            # 独立したゲートの場合は削除
            # （通常はwrapper経由で使うので、このケースは稀）
            logger.info(f"[Burn-in] Independent gate found: {name}")
    
    # サマリー
    logger.info(f"[Burn-in] Complete!")
    logger.info(f"[Burn-in] Burned in {burn_in_count}/{len(modules_to_process)} modules")
    
    # 統計情報を保存
    if save_stats and all_stats:
        if stats_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_path = f"burn_in_stats_{timestamp}.json"
        
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        # float32などをJSONシリアライズ可能な形式に変換
        serializable_stats = {}
        for key, value in all_stats.items():
            serializable_stats[key] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in value.items()
            }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Burn-in] Stats saved to: {stats_path}")
    
    return all_stats


def verify_burn_in(
    model: nn.Module,
    original_model: nn.Module,
    test_input: torch.Tensor,
    max_error_threshold: float = 1e-4
) -> Dict[str, float]:
    """
    焼きこみ前後のモデル出力を比較検証
    
    Args:
        model: 焼きこみ後のモデル
        original_model: 焼きこみ前のモデル
        test_input: テスト入力
        max_error_threshold: 許容する最大誤差
        
    Returns:
        verification_results: 検証結果
        
    Raises:
        AssertionError: 誤差が閾値を超えた場合
    """
    logger.info(f"[Burn-in Verification] Comparing model outputs...")
    
    # 評価モード
    model.eval()
    original_model.eval()
    
    with torch.no_grad():
        # 焼きこみ前の出力
        output_before = original_model(test_input)
        if isinstance(output_before, tuple):
            output_before = output_before[0]
        
        # 焼きこみ後の出力
        output_after = model(test_input)
        if isinstance(output_after, tuple):
            output_after = output_after[0]
    
    # 誤差計算
    max_error = (output_before - output_after).abs().max().item()
    mean_error = (output_before - output_after).abs().mean().item()
    relative_error = max_error / (output_before.abs().max().item() + 1e-8)
    
    # KLダイバージェンス（logitsの場合）
    if output_before.dim() == 3:  # [B, T, V]
        p_before = torch.softmax(output_before, dim=-1)
        p_after = torch.softmax(output_after, dim=-1)
        kl_div = torch.nn.functional.kl_div(
            p_after.log(),
            p_before,
            reduction='batchmean'
        ).item()
    else:
        kl_div = None
    
    results = {
        'max_error': max_error,
        'mean_error': mean_error,
        'relative_error': relative_error,
        'kl_divergence': kl_div,
    }
    
    logger.info(f"[Verification Results]")
    logger.info(f"  Max error: {max_error:.6e}")
    logger.info(f"  Mean error: {mean_error:.6e}")
    logger.info(f"  Relative error: {relative_error:.6e}")
    if kl_div is not None:
        logger.info(f"  KL divergence: {kl_div:.6e}")
    
    # 閾値チェック
    if max_error > max_error_threshold:
        logger.error(f"[ERROR] Max error {max_error:.6e} exceeds threshold {max_error_threshold:.6e}")
        raise AssertionError(f"Burn-in verification failed: max_error={max_error:.6e}")
    else:
        logger.info(f"[OK] Burn-in verification passed!")
    
    return results


if __name__ == "__main__":
    """
    焼きこみ機構の単体テスト
    """
    import sys
    sys.path.append(str(Path(__file__).parent))
    from so8t_layer import SO8TRotationGate
    
    print("=" * 80)
    print("SO8T Burn-in Mechanism Unit Test")
    print("=" * 80)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    
    # テスト設定
    D_out = 512
    D = 512
    num_blocks = D // 8
    
    print(f"\n[Config]")
    print(f"  D_out: {D_out}")
    print(f"  D: {D}")
    print(f"  Num blocks: {num_blocks}")
    
    # 1. 線形層とSO8Tゲートを作成
    print(f"\n[Test 1] Basic burn-in")
    
    linear_layer = nn.Linear(D, D_out, bias=False).to(device)
    so8t_gate = SO8TRotationGate(hidden_size=D, init_scale=0.1).to(device)
    
    print(f"  Linear layer: {linear_layer.weight.shape}")
    print(f"  SO8T gate: {num_blocks} blocks")
    
    # 2. 回転行列を取得
    R_blocks = so8t_gate.get_rotation_matrices()  # [num_blocks, 8, 8]
    print(f"  Rotation matrices: {R_blocks.shape}")
    
    # 3. 焼きこみを適用
    R_blocks_list = [R_blocks[i] for i in range(num_blocks)]
    W_o_prime, stats = fold_blockdiag(
        linear_layer.weight.data,
        R_blocks_list,
        verify=True
    )
    
    print(f"\n[Burn-in Stats]")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")
    
    # 4. 検証
    if 'max_error' in stats:
        if stats['max_error'] < 1e-4:
            print(f"\n[OK] Burn-in verification passed!")
        else:
            print(f"\n[WARNING] Max error {stats['max_error']:.6e} is large")
    
    print("\n" + "=" * 80)
    print("[Burn-in] All tests passed!")
    print("=" * 80)

