#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8Tモデルローダー

SO8Tモデルロードの統一ユーティリティ、複数のモデルパスからの自動検出、フォールバック処理
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))

# Transformersライブラリのインポート
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available")

# SO8Tモデルのインポート
try:
    from so8t_mmllm.src.models.safety_aware_so8t import SafetyAwareSO8TConfig
    from so8t_mmllm.src.models.so8t_thinking_model import SO8TThinkingModel
    SO8T_MODEL_AVAILABLE = True
except ImportError:
    SO8T_MODEL_AVAILABLE = False
    logging.warning("SO8T model modules not available")

# ロギング設定
logger = logging.getLogger(__name__)


def find_so8t_model_paths() -> List[Path]:
    """
    利用可能なSO8Tモデルパスを検索
    
    Returns:
        見つかったモデルパスのリスト
    """
    default_paths = [
        Path("D:/webdataset/models/so8t-phi4-so8t-ja-finetuned"),
        Path("D:/webdataset/models/so8t-phi3-so8t-ja-finetuned"),
        Path("models/so8t-phi4-so8t-ja-finetuned"),
        Path("models/so8t-phi3-so8t-ja-finetuned"),
        Path("so8t-mmllm/models/so8t-phi4-so8t-ja-finetuned"),
        Path("so8t-mmllm/models/so8t-phi3-so8t-ja-finetuned"),
        Path(PROJECT_ROOT / "models" / "so8t-phi4-so8t-ja-finetuned"),
        Path(PROJECT_ROOT / "models" / "so8t-phi3-so8t-ja-finetuned"),
    ]
    
    found_paths = []
    
    for path in default_paths:
        # 絶対パスに変換
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        
        # モデルディレクトリまたはconfig.jsonファイルの存在確認
        if path.exists():
            # ディレクトリの場合
            if path.is_dir():
                config_file = path / "config.json"
                if config_file.exists():
                    found_paths.append(path)
                    logger.debug(f"[SO8T] Found model at: {path}")
            # ファイルの場合（config.jsonなど）
            elif path.suffix == '.json' and path.name == 'config.json':
                found_paths.append(path.parent)
                logger.debug(f"[SO8T] Found model at: {path.parent}")
    
    return found_paths


def validate_so8t_model(model_path: Path) -> bool:
    """
    SO8Tモデルの有効性を検証
    
    Args:
        model_path: モデルパス
    
    Returns:
        検証成功フラグ
    """
    if not model_path.exists():
        return False
    
    # config.jsonの存在確認
    config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
    if not config_file.exists():
        logger.debug(f"[SO8T] config.json not found: {config_file}")
        return False
    
    # tokenizer.jsonまたはtokenizer_config.jsonの存在確認
    tokenizer_files = [
        model_path / "tokenizer.json",
        model_path / "tokenizer_config.json",
        model_path / "vocab.json"
    ]
    
    has_tokenizer = any(f.exists() for f in tokenizer_files)
    if not has_tokenizer:
        logger.debug(f"[SO8T] Tokenizer files not found in: {model_path}")
        return False
    
    logger.debug(f"[SO8T] Model validation passed: {model_path}")
    return True


def load_so8t_model(
    model_path: Optional[str] = None,
    device: str = "auto",
    use_quadruple_thinking: bool = True,
    use_redacted_tokens: bool = False,
    fallback_to_default: bool = True
) -> Tuple[Optional[Any], Optional[Any], bool]:
    """
    SO8Tモデルをロード（自動パス検出）
    
    Args:
        model_path: モデルパス（Noneの場合は自動検出）
        device: デバイス（"auto", "cuda", "cpu"）
        use_quadruple_thinking: 四重推論を使用するか
        use_redacted_tokens: 編集済みトークンを使用するか
        fallback_to_default: フォールバック処理を使用するか
    
    Returns:
        (model, tokenizer, success) のタプル
    """
    if not SO8T_MODEL_AVAILABLE:
        logger.error("[SO8T] SO8T model modules not available")
        if fallback_to_default:
            return _load_fallback_model(device)
        return None, None, False
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("[SO8T] transformers not available")
        if fallback_to_default:
            return _load_fallback_model(device)
        return None, None, False
    
    # モデルパスの検出
    actual_model_path = None
    
    if model_path:
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path_obj = PROJECT_ROOT / model_path_obj
        
        if validate_so8t_model(model_path_obj):
            actual_model_path = model_path_obj
            logger.info(f"[SO8T] Using specified model path: {actual_model_path}")
        else:
            logger.warning(f"[SO8T] Specified model path invalid: {model_path}")
    
    # 自動検出
    if actual_model_path is None:
        found_paths = find_so8t_model_paths()
        
        for path in found_paths:
            if validate_so8t_model(path):
                actual_model_path = path
                logger.info(f"[SO8T] Auto-detected model path: {actual_model_path}")
                break
    
    # モデルパスが見つからない場合
    if actual_model_path is None:
        logger.warning("[SO8T] No valid SO8T model found")
        if fallback_to_default:
            logger.info("[SO8T] Falling back to default model")
            return _load_fallback_model(device)
        return None, None, False
    
    # モデルをロード
    try:
        logger.info(f"[SO8T] Loading model from: {actual_model_path}")
        
        # トークナイザーを読み込み
        tokenizer = AutoTokenizer.from_pretrained(str(actual_model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # SO8T設定
        so8t_config = SafetyAwareSO8TConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=11008,
            max_position_embeddings=4096,
            use_so8_rotation=True,
            use_safety_head=True,
            use_verifier_head=True
        )
        
        # SO8TThinkingModelを読み込み
        model = SO8TThinkingModel(
            base_model_name_or_path=str(actual_model_path),
            so8t_config=so8t_config,
            use_redacted_tokens=use_redacted_tokens,
            use_quadruple_thinking=use_quadruple_thinking
        )
        
        # トークナイザーを設定
        model.set_tokenizer(tokenizer)
        
        # 評価モードに設定
        model.eval()
        
        # デバイスに移動
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device != "cpu":
            try:
                model = model.to(device)
            except Exception as e:
                logger.warning(f"[SO8T] Failed to move model to {device}: {e}, using CPU")
                device = "cpu"
                model = model.to(device)
        else:
            model = model.to(device)
        
        logger.info(f"[SO8T] Model loaded successfully on {device}")
        return model, tokenizer, True
    
    except Exception as e:
        logger.error(f"[SO8T] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        
        if fallback_to_default:
            logger.info("[SO8T] Falling back to default model")
            return _load_fallback_model(device)
        
        return None, None, False


def _load_fallback_model(device: str) -> Tuple[Optional[Any], Optional[Any], bool]:
    """
    フォールバックモデルをロード（デフォルトのPhi-3モデル）
    
    Args:
        device: デバイス
    
    Returns:
        (model, tokenizer, success) のタプル
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None, False
    
    try:
        logger.info("[SO8T] Loading fallback model: microsoft/Phi-3-mini-4k-instruct")
        
        from transformers import AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype="auto",
            device_map=device if device != "auto" else "auto"
        )
        
        model.eval()
        
        logger.info("[SO8T] Fallback model loaded successfully")
        return model, tokenizer, True
    
    except Exception as e:
        logger.error(f"[SO8T] Failed to load fallback model: {e}")
        return None, None, False


def get_so8t_model_info(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    SO8Tモデルの情報を取得
    
    Args:
        model_path: モデルパス（Noneの場合は自動検出）
    
    Returns:
        モデル情報の辞書
    """
    info = {
        'model_path': None,
        'is_valid': False,
        'has_tokenizer': False,
        'has_config': False,
        'available_paths': []
    }
    
    # 利用可能なパスを検索
    found_paths = find_so8t_model_paths()
    info['available_paths'] = [str(p) for p in found_paths]
    
    # 指定されたパスまたは最初の有効なパスを検証
    if model_path:
        path_to_check = Path(model_path)
    elif found_paths:
        path_to_check = found_paths[0]
    else:
        return info
    
    if not path_to_check.is_absolute():
        path_to_check = PROJECT_ROOT / path_to_check
    
    info['model_path'] = str(path_to_check)
    info['is_valid'] = validate_so8t_model(path_to_check)
    
    if path_to_check.exists():
        config_file = path_to_check / "config.json"
        info['has_config'] = config_file.exists()
        
        tokenizer_files = [
            path_to_check / "tokenizer.json",
            path_to_check / "tokenizer_config.json"
        ]
        info['has_tokenizer'] = any(f.exists() for f in tokenizer_files)
    
    return info

