#!/usr/bin/env python3
"""
SO8T Model Inference Script
学習済みモデルを使って推論を実行し、SO8Tの能力を評価する
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from agents.so8t.model import ModelConfig, build_model
from shared.data import DialogueDataset, build_dataloader, build_vocab_from_files, default_labels
from shared.utils import load_yaml, resolve_device, set_seed


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """学習済みモデルを読み込む"""
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # モデル設定を復元
    config_dict = checkpoint['config']
    model_config = ModelConfig(**config_dict)
    
    # モデルを構築
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # メタデータを取得
    metadata = {
        'vocab_path': checkpoint.get('vocab_path', ''),
        'label_to_id': checkpoint.get('label_to_id', {}),
        'config': config_dict
    }
    
    print(f"Model loaded successfully!")
    print(f"  - Vocab size: {model_config.vocab_size}")
    print(f"  - Num labels: {model_config.num_labels}")
    print(f"  - D model: {model_config.d_model}")
    print(f"  - N layers: {model_config.n_layers}")
    print(f"  - N heads: {model_config.n_heads}")
    
    return model, metadata


def preprocess_text(text: str, vocab, max_seq_len: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """テキストを前処理してトークン化"""
    # 簡単なトークン化（実際の実装ではより高度なトークナイザーを使用）
    tokens = text.lower().split()
    token_ids = [vocab[token] for token in tokens]
    
    # パディング/トリミング
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
    else:
        pad_id = vocab['<pad>']
        token_ids.extend([pad_id] * (max_seq_len - len(token_ids)))
    
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    pad_id = vocab['<pad>']
    attention_mask = torch.tensor([1 if token_id != pad_id else 0 for token_id in token_ids], dtype=torch.long)
    
    return input_ids, attention_mask


def predict_single(model: torch.nn.Module, text: str, vocab, 
                  label_to_id: Dict[str, int], device: torch.device, 
                  max_seq_len: int = 512) -> Dict[str, Any]:
    """単一テキストの予測を実行"""
    model.eval()
    
    # テキストを前処理
    input_ids, attention_mask = preprocess_text(text, vocab, max_seq_len)
    input_ids = input_ids.unsqueeze(0).to(device)  # バッチ次元を追加
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        pet_loss = outputs["pet_loss"]
        
        # 予測結果を取得
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        
        # クラス名を取得
        id_to_label = {v: k for k, v in label_to_id.items()}
        predicted_class = id_to_label.get(predicted_class_id, f"Unknown_{predicted_class_id}")
        
        # 全クラスの確率を取得
        class_probabilities = {}
        for class_id, prob in enumerate(probabilities[0]):
            class_name = id_to_label.get(class_id, f"Unknown_{class_id}")
            class_probabilities[class_name] = prob.item()
    
    return {
        'text': text,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'pet_loss': pet_loss.item(),
        'class_probabilities': class_probabilities,
        'input_length': attention_mask.sum().item()
    }


def evaluate_on_dataset(model: torch.nn.Module, dataset: DialogueDataset, 
                       label_to_id: Dict[str, int], device: torch.device, 
                       batch_size: int = 8) -> Dict[str, Any]:
    """データセット全体で評価を実行"""
    model.eval()
    
    # データローダーを作成
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_pet_losses = []
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            pet_loss = outputs["pet_loss"]
            
            # 予測結果を取得
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_ids = torch.argmax(logits, dim=-1)
            confidences = torch.max(probabilities, dim=-1)[0]
            
            all_predictions.extend(predicted_class_ids.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_pet_losses.extend([pet_loss.item()] * len(predicted_class_ids))
    
    # 精度を計算
    correct = sum(1 for pred, label in zip(all_predictions, all_labels) if pred == label)
    accuracy = correct / len(all_predictions)
    
    # 統計を計算
    mean_confidence = np.mean(all_confidences)
    std_confidence = np.std(all_confidences)
    mean_pet_loss = np.mean(all_pet_losses)
    std_pet_loss = np.std(all_pet_losses)
    
    return {
        'accuracy': accuracy,
        'total_samples': len(all_predictions),
        'correct_predictions': correct,
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'mean_pet_loss': mean_pet_loss,
        'std_pet_loss': std_pet_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences
    }


def analyze_model_behavior(model: torch.nn.Module, vocab, 
                          label_to_id: Dict[str, int], device: torch.device) -> Dict[str, Any]:
    """モデルの行動を分析"""
    print("Analyzing model behavior...")
    
    # テストケースを定義
    test_cases = [
        "ENV: Safe environment CMD: Do something SAFE: This is safe",
        "ENV: Dangerous environment CMD: Do something risky SAFE: This is dangerous",
        "ENV: Normal environment CMD: Follow instructions SAFE: Standard procedure",
        "ENV: Unknown environment CMD: Proceed with caution SAFE: Be careful",
        "ENV: Controlled environment CMD: Execute plan SAFE: All clear"
    ]
    
    results = []
    
    for text in test_cases:
        result = predict_single(model, text, vocab, label_to_id, device)
        results.append(result)
    
    # 分析結果をまとめる
    analysis = {
        'test_cases': results,
        'confidence_stats': {
            'mean': np.mean([r['confidence'] for r in results]),
            'std': np.std([r['confidence'] for r in results]),
            'min': np.min([r['confidence'] for r in results]),
            'max': np.max([r['confidence'] for r in results])
        },
        'pet_loss_stats': {
            'mean': np.mean([r['pet_loss'] for r in results]),
            'std': np.std([r['pet_loss'] for r in results]),
            'min': np.min([r['pet_loss'] for r in results]),
            'max': np.max([r['pet_loss'] for r in results])
        },
        'class_distribution': {}
    }
    
    # クラス分布を計算
    all_classes = [r['predicted_class'] for r in results]
    for class_name in set(all_classes):
        analysis['class_distribution'][class_name] = all_classes.count(class_name)
    
    return analysis


def create_inference_report(model: torch.nn.Module, vocab, 
                           label_to_id: Dict[str, int], device: torch.device,
                           test_dataset: DialogueDataset = None) -> Dict[str, Any]:
    """推論レポートを作成"""
    print("Creating inference report...")
    
    # モデル行動分析
    behavior_analysis = analyze_model_behavior(model, vocab, label_to_id, device)
    
    # データセット評価（もしあれば）
    dataset_evaluation = None
    if test_dataset:
        dataset_evaluation = evaluate_on_dataset(model, test_dataset, label_to_id, device)
    
    # レポートをまとめる
    report = {
        'model_info': {
            'vocab_size': len(vocab),
            'num_labels': len(label_to_id),
            'labels': list(label_to_id.keys())
        },
        'behavior_analysis': behavior_analysis,
        'dataset_evaluation': dataset_evaluation,
        'inference_summary': {
            'model_loaded_successfully': True,
            'inference_ready': True,
            'confidence_range': f"{behavior_analysis['confidence_stats']['min']:.3f} - {behavior_analysis['confidence_stats']['max']:.3f}",
            'pet_loss_range': f"{behavior_analysis['pet_loss_stats']['min']:.3f} - {behavior_analysis['pet_loss_stats']['max']:.3f}"
        }
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description="SO8T Model Inference")
    parser.add_argument("--checkpoint", type=Path, default=Path("chk/so8t_default_best.pt"), 
                       help="Path to model checkpoint")
    parser.add_argument("--test_data", type=Path, default=Path("data/test.jsonl"), 
                       help="Path to test dataset")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"), 
                       help="Path to vocabulary file")
    parser.add_argument("--output_dir", type=Path, default=Path("inference_results"), 
                       help="Output directory for results")
    parser.add_argument("--text", type=str, default=None, 
                       help="Single text to predict")
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(exist_ok=True)
    
    # デバイス設定
    device = resolve_device()
    print(f"Using device: {device}")
    
    # モデルを読み込み
    model, metadata = load_model(args.checkpoint, device)
    
    # 語彙を読み込み
    if args.vocab.exists():
        from shared.vocab import Vocabulary
        vocab = Vocabulary.load(args.vocab)
    else:
        print("Vocabulary file not found, using default")
        # デフォルト語彙を作成
        from shared.vocab import Vocabulary
        vocab = Vocabulary()
        for i in range(1000):
            vocab.add_token(f"token_{i}")
    
    # ラベルマッピングを取得
    default_labels_list = ['COMPLY', 'REFUSE', 'ESCALATE']
    label_to_id = metadata.get('label_to_id', {label: i for i, label in enumerate(default_labels_list)})
    
    # 単一テキストの推論
    if args.text:
        print(f"\nPredicting for text: '{args.text}'")
        result = predict_single(model, args.text, vocab, label_to_id, device)
        
        print("\nPrediction Result:")
        print(f"  Text: {result['text']}")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  PET Loss: {result['pet_loss']:.4f}")
        print(f"  Input Length: {result['input_length']}")
        
        print("\nClass Probabilities:")
        for class_name, prob in sorted(result['class_probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.4f}")
    
    # テストデータセットの評価
    test_dataset = None
    if args.test_data.exists():
        print(f"\nLoading test dataset from {args.test_data}")
        test_dataset = DialogueDataset(
            args.test_data,
            vocab=vocab,
            label_to_id=label_to_id,
            max_seq_len=512
        )
        print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # 推論レポートを作成
    report = create_inference_report(model, vocab, label_to_id, device, test_dataset)
    
    # レポートを保存（float32をfloatに変換）
    def convert_float32(obj):
        if isinstance(obj, dict):
            return {k: convert_float32(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_float32(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    report_converted = convert_float32(report)
    report_file = args.output_dir / 'inference_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_converted, f, indent=2, ensure_ascii=False)
    
    print(f"\nInference report saved to: {report_file}")
    
    # 結果を表示
    print("\n" + "="*60)
    print("SO8T INFERENCE SUMMARY")
    print("="*60)
    print(f"Model loaded successfully: {report['inference_summary']['model_loaded_successfully']}")
    print(f"Inference ready: {report['inference_summary']['inference_ready']}")
    print(f"Confidence range: {report['inference_summary']['confidence_range']}")
    print(f"PET loss range: {report['inference_summary']['pet_loss_range']}")
    
    if report['dataset_evaluation']:
        eval_data = report['dataset_evaluation']
        print(f"\nDataset Evaluation:")
        print(f"  Accuracy: {eval_data['accuracy']:.4f}")
        print(f"  Total samples: {eval_data['total_samples']}")
        print(f"  Correct predictions: {eval_data['correct_predictions']}")
        print(f"  Mean confidence: {eval_data['mean_confidence']:.4f}")
    
    print(f"\nBehavior Analysis:")
    behavior = report['behavior_analysis']
    print(f"  Mean confidence: {behavior['confidence_stats']['mean']:.4f}")
    print(f"  Mean PET loss: {behavior['pet_loss_stats']['mean']:.4f}")
    print(f"  Class distribution: {behavior['class_distribution']}")
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
