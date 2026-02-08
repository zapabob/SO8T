"""
NSFW・危険コンテンツの統計的ラベリング

scikit-learnを使用して、小規模な手動ラベル付きデータから
大規模コーパスを自動ラベリングする。
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ラベル定義
LABEL_CATEGORIES = [
    "safe",
    "nsfw",
    "violence",
    "harassment",
    "self_harm",
    "weapons_detail",  # 具体的手順は拒否
    "medical_advice_high_risk",
    "illegal_content",
]


class NSFWLabeler:
    """NSFW/危険コンテンツラベラー"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Args:
            model_path: 保存済みモデルのパス（オプション）
        """
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.classifier = None
        self.model_path = model_path
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def load_labeled_data(self, file_path: Path) -> tuple[List[str], List[str]]:
        """
        ラベル付きデータをロード
        
        Args:
            file_path: JSONLファイルパス
        
        Returns:
            (texts, labels) のタプル
        """
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                sample = json.loads(line)
                text = sample.get("text", "")
                label = sample.get("label", "safe")
                
                if text and label in LABEL_CATEGORIES:
                    texts.append(text)
                    labels.append(label)
        
        return texts, labels
    
    def train(self, labeled_file: Path, test_size: float = 0.2, random_state: int = 42):
        """
        分類器を訓練
        
        Args:
            labeled_file: ラベル付きデータファイル
            test_size: テストセットの割合
            random_state: 乱数シード
        """
        print(f"[INFO] Loading labeled data from: {labeled_file}")
        texts, labels = self.load_labeled_data(labeled_file)
        
        if len(texts) == 0:
            raise ValueError("No labeled data found")
        
        print(f"[INFO] Loaded {len(texts)} labeled samples")
        
        # ベクトル化
        print("[INFO] Vectorizing texts...")
        X = self.vectorizer.fit_transform(texts)
        
        # 訓練/テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, stratify=labels, random_state=random_state
        )
        
        # 分類器を訓練
        print("[INFO] Training classifier...")
        self.classifier = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
        )
        self.classifier.fit(X_train, y_train)
        
        # 評価
        print("[INFO] Evaluating...")
        y_pred = self.classifier.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, text: str) -> tuple[str, float]:
        """
        テキストのラベルを予測
        
        Args:
            text: 予測対象のテキスト
        
        Returns:
            (predicted_label, confidence) のタプル
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        predicted_label = self.classifier.classes_[pred_idx]
        
        return predicted_label, float(confidence)
    
    def auto_label_dataset(
        self,
        input_file: Path,
        output_file: Path,
        confidence_threshold: float = 0.9,
    ) -> int:
        """
        データセットを自動ラベリング
        
        Args:
            input_file: 入力データセット（JSONL）
            output_file: 出力データセット（JSONL、ラベル追加）
            confidence_threshold: 信頼度閾値（この値以上のみ採用）
        
        Returns:
            ラベル付きサンプル数
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        labeled_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                sample = json.loads(line)
                text = sample.get("content", sample.get("text", ""))
                
                if not text:
                    continue
                
                # 予測
                pred_label, confidence = self.predict(text)
                
                # 信頼度が閾値以上の場合のみ採用
                if confidence >= confidence_threshold:
                    sample["nsfw_label"] = pred_label
                    sample["nsfw_confidence"] = confidence
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    labeled_count += 1
        
        return labeled_count
    
    def save_model(self, model_path: Path):
        """モデルを保存"""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
            },
            model_path,
        )
        print(f"[INFO] Model saved to: {model_path}")
    
    def load_model(self, model_path: Path):
        """モデルをロード"""
        model_data = joblib.load(model_path)
        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        print(f"[INFO] Model loaded from: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="NSFW/危険コンテンツラベリング")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "label"],
        required=True,
        help="Mode: train or label",
    )
    parser.add_argument(
        "--labeled-data",
        type=Path,
        help="Labeled data file (for train mode)",
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        help="Input data file (for label mode)",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        help="Output data file (for label mode)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/nsfw_classifier.joblib"),
        help="Model path",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for auto-labeling",
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.labeled_data:
            parser.error("--labeled-data is required for train mode")
        
        labeler = NSFWLabeler()
        labeler.train(args.labeled_data)
        labeler.save_model(args.model_path)
    
    elif args.mode == "label":
        if not args.input_data or not args.output_data:
            parser.error("--input-data and --output-data are required for label mode")
        
        labeler = NSFWLabeler(model_path=args.model_path)
        if labeler.classifier is None:
            parser.error(f"Model not found at {args.model_path}. Train first.")
        
        count = labeler.auto_label_dataset(
            args.input_data,
            args.output_data,
            confidence_threshold=args.confidence_threshold,
        )
        print(f"[SUCCESS] Labeled {count} samples")


if __name__ == "__main__":
    main()

