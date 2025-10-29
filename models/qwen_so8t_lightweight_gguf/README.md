# SO8T Lightweight Distilled Model

## 概要
SO8T-Phi31-Mini-128K-Enhanced-Q8_0.ggufから知識蒸留により作成された軽量Transformerモデルです。

## 特徴
- **知識蒸留**: 大規模モデルから軽量モデルへの効率的な知識転移
- **高圧縮率**: 約73%のパラメータ削減を実現
- **高速推論**: 軽量構造による高速な推論実行
- **重み安定性**: 重み崩壊を防ぐ高度な安定化技術

## 蒸留情報
- **教師モデル**: SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf
- **蒸留方法**: 温度付きKL divergence損失
- **圧縮率**: 73%
- **最終損失**: 0.281206
- **学習エポック**: 5

## モデル仕様
- **アーキテクチャ**: SimpleStudentModel
- **モデルタイプ**: distilled_transformer
- **語彙サイズ**: 32,000
- **隠れサイズ**: 512
- **中間サイズ**: 2,048
- **レイヤー数**: 4
- **アテンションヘッド数**: 8
- **最大位置埋め込み**: 1,024

## パラメータ統計
- **総パラメータ数**: 45,933,824
- **学習可能パラメータ数**: 45,933,824
- **モデルサイズ**: 0.17 GB (float32)

## 使用方法
```python
import torch
from models.lightweight_model import SimpleStudentModel

# モデル読み込み
model = SimpleStudentModel(vocab_size=32000, hidden_size=512, num_layers=4)
checkpoint = torch.load('lightweight_model_weights.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 推論実行
input_ids = torch.randint(0, 32000, (1, 64))
outputs = model(input_ids)
```

## ライセンス
Apache-2.0

## 作成者
SO8T Safe Agent Project

## 作成日
2025-10-29 06:45:04
