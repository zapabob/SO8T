#!/usr/bin/env python3
"""Unslothモデルの構造をデバッグ"""

import torch
from unsloth import FastLanguageModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_unsloth_model.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def debug_model_structure():
    """Unslothモデルの構造をデバッグ"""
    try:
        logger.info("Loading model with Unsloth for debugging...")
        print("Starting model loading...")

        model_path = "models/Borea-Phi-3.5-mini-Instruct-Jp"

        # Unslothでモデルをロード
        print(f"Loading model from {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
        )
        print("Model loaded successfully")

        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model class: {model.__class__.__name__}")

        # モデルの属性を調査
        logger.info("Model attributes:")
        for attr in dir(model):
            if not attr.startswith('_'):
                try:
                    value = getattr(model, attr)
                    if hasattr(value, '__len__') and len(value) > 0:
                        logger.info(f"  {attr}: {type(value)} (length: {len(value)})")
                    else:
                        logger.info(f"  {attr}: {type(value)}")
                except:
                    logger.info(f"  {attr}: <error accessing>")

        # layers属性があるか確認
        if hasattr(model, 'model'):
            logger.info("model.model exists")
            if hasattr(model.model, 'layers'):
                logger.info(f"model.model.layers exists, length: {len(model.model.layers)}")
                for i, layer in enumerate(model.model.layers[:3]):  # 最初の3層のみ
                    logger.info(f"  Layer {i}: {type(layer)}")
            else:
                logger.info("model.model.layers does not exist")
                logger.info("Available attributes in model.model:")
                for attr in dir(model.model):
                    if not attr.startswith('_'):
                        logger.info(f"    {attr}")
        else:
            logger.info("model.model does not exist")

    except Exception as e:
        logger.error(f"Failed to debug model structure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_structure()
