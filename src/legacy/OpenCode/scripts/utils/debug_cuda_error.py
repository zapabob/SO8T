#!/usr/bin/env python3
"""
CUDAエラーのデバッグスクリプト
"""

import torch
from pathlib import Path
from shared.data import DialogueDataset, collate_batch
from shared.vocab import Vocabulary
from agents.so8t.model_safety import build_safety_model, SafetyModelConfig

def debug_cuda_error():
    print("Loading data...")
    vocab = Vocabulary.from_file(Path('data/vocab.json'))
    
    dataset = DialogueDataset(
        path=Path('data/train.jsonl'),
        vocab=vocab,
        label_to_id={"COMPLY": 0, "REFUSE": 1, "ESCALATE": 2},
        max_seq_len=512
    )
    
    def collate_fn(batch):
        return collate_batch(batch, pad_index=vocab.pad_index)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print("Building model...")
    model_config = SafetyModelConfig(
        vocab_size=len(vocab),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        num_labels=3,
        num_safety_labels=3,
        max_seq_len=512,
        gate_order=["R_env", "R_safe", "R_cmd"],
        safety_first=True
    )
    
    model = build_safety_model(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model token embeddings shape: {model.token_embeddings.weight.shape}")
    print(f"Vocab size: {len(vocab)}")
    
    # 最初のバッチを取得
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Input IDs max: {batch['input_ids'].max().item()}")
    print(f"Input IDs min: {batch['input_ids'].min().item()}")
    print(f"Input IDs: {batch['input_ids'].tolist()}")
    
    # デバイスに移動
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    print(f"Input IDs on device max: {input_ids.max().item()}")
    print(f"Input IDs on device min: {input_ids.min().item()}")
    
    # モデルを評価モードに
    model.eval()
    
    print("Testing model forward pass...")
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print("SUCCESS: Model forward pass completed!")
        print(f"Output keys: {outputs.keys()}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Error type: {type(e)}")
        
        # より詳細なデバッグ
        print("\nDetailed debugging:")
        print(f"Input IDs range: {input_ids.min().item()} to {input_ids.max().item()}")
        print(f"Token embeddings vocab size: {model.token_embeddings.weight.shape[0]}")
        print(f"Input IDs unique values: {torch.unique(input_ids).tolist()}")
        
        # 問題のあるインデックスを特定
        invalid_indices = input_ids >= model.token_embeddings.weight.shape[0]
        if invalid_indices.any():
            print(f"Invalid indices found: {input_ids[invalid_indices].tolist()}")
            print(f"Invalid positions: {torch.where(invalid_indices)}")

if __name__ == "__main__":
    debug_cuda_error()
