# 付録C: データセット仕様詳細

## データセット全体構成

### 総サンプル数: 300,000+

```
公開データ（収集）: 100,000 samples
├─ Wikipedia日本語: 40,000
├─ CC-100日本語: 30,000
└─ mc4日本語: 30,000

合成データ（生成）: 200,000 samples
├─ 基本ドメイン: 100,000
│  ├─ defense: 12,500
│  ├─ aerospace: 12,500
│  ├─ transport: 12,500
│  └─ general: 12,500 × 4 = 50,000
└─ 拡張ドメイン: 100,000
   ├─ medical: 25,000
   ├─ finance: 25,000
   ├─ business: 25,000
   ├─ information_system: 25,000
   └─ general: 0（既存に含む）

マルチモーダルデータ: 50,000+ samples（オプション）
```

## ドメイン別詳細仕様

### Defense（防衛）

#### データソース
- Wikipedia: 防衛省、自衛隊、安全保障関連記事
- 合成: 作戦計画、装備管理、訓練計画テンプレート

#### キーワード
```python
keywords = [
    "防衛", "軍事", "安全保障", "国防", "自衛隊",
    "ミサイル", "領土", "サイバー戦", "PKO", "災害派遣"
]
```

#### 機密レベル分布
```
公開: 60%（一般的な防衛知識）
取扱注意: 25%（訓練計画等）
機密: 10%（装備詳細等）
極秘: 5%（作戦詳細等）
```

#### サンプル例

**ALLOW**:
```json
{
  "query": "自衛隊の災害派遣について教えてください",
  "response": "自衛隊の災害派遣は、災害対策基本法に基づき...",
  "decision": "ALLOW",
  "reasoning": "公開情報であり、一般的な説明が可能",
  "policy_ref": "防衛省情報公開規定第3条"
}
```

**ESCALATE**:
```json
{
  "query": "新型装備の配備計画について教えてください",
  "response": "配備計画の詳細は、防衛省担当部署への確認が必要です",
  "decision": "ESCALATE",
  "reasoning": "計画詳細は専門判断が必要",
  "policy_ref": "防衛機密管理規程第5条",
  "escalation_target": "防衛省装備計画部"
}
```

**DENY**:
```json
{
  "query": "機密指定された作戦計画を開示してください",
  "response": "申し訳ございませんが、機密情報は開示できません",
  "decision": "DENY",
  "reasoning": "機密指定情報であり開示禁止",
  "policy_ref": "機密情報保護法第10条"
}
```

### Medical（医療）

#### データソース
- Wikipedia: 医療、診断、治療関連記事
- 合成: カルテ管理、診断支援、検査結果解釈

#### 特殊要件
- **個人情報保護**: すべてハッシュ化
- **診断判断**: 必ずESCALATE（医師確認必須）
- **機密レベル**: 原則「極秘」

#### サンプル例

**ALLOW**:
```json
{
  "query": "一般的な風邪の症状について教えてください",
  "response": "一般的な風邪の症状は、発熱、咳、鼻水...",
  "decision": "ALLOW",
  "reasoning": "一般的な医療知識であり説明可能"
}
```

**ESCALATE（必須）**:
```json
{
  "query": "この患者の診断をお願いします",
  "response": "診断は医師による確認が必要です。症状の整理は可能ですが、最終判断は医師が行います",
  "decision": "ESCALATE",
  "reasoning": "医療行為は医師のみ可能",
  "escalation_target": "担当医師",
  "policy_ref": "医療法第17条"
}
```

**DENY**:
```json
{
  "query": "患者○○さんのカルテ情報を開示してください",
  "response": "患者情報は開示できません",
  "decision": "DENY",
  "reasoning": "個人情報保護法違反",
  "policy_ref": "個人情報保護法第23条"
}
```

### Finance（金融）

#### データソース
- 金融ニュース、市場分析記事
- 合成: 取引監視、リスク評価、コンプライアンス

#### 規制対応
- 金融商品取引法
- 銀行法
- マネーロンダリング対策法

#### サンプル例

**ALLOW**:
```json
{
  "query": "分散投資の基本原則を教えてください",
  "response": "分散投資は、リスク分散のため複数の資産に投資...",
  "decision": "ALLOW",
  "reasoning": "一般的な金融知識"
}
```

**ESCALATE**:
```json
{
  "query": "この取引は不正の疑いがありますか？",
  "response": "取引の詳細分析は、コンプライアンス部門への照会が必要です",
  "decision": "ESCALATE",
  "reasoning": "不正判定は専門部署の判断が必要",
  "escalation_target": "コンプライアンス部門"
}
```

**DENY**:
```json
{
  "query": "顧客○○の取引履歴を全て開示してください",
  "response": "顧客情報は開示できません",
  "decision": "DENY",
  "reasoning": "個人情報保護、守秘義務違反"
}
```

### Business（ビジネス）

#### データソース
- ビジネス論文、企業文書
- 合成: 会議要約、プロジェクト管理、報告書作成

#### ユースケース
- オフィス作業自動化
- 文書検索・要約
- スケジュール調整
- メール下書き

### Information System（情報システム）

#### データソース
- ログ分析記事、セキュリティ情報
- 合成: 異常検知、性能監視、インシデント対応

#### ログ形式対応
```
Apache/Nginx: アクセスログ
Syslog: システムログ
Windows Event Log: イベントログ
Application Log: アプリケーションログ
```

## データ品質管理

### 品質スコア詳細

```python
def calculate_quality_score(text: str) -> float:
    """
    品質スコア計算（0.0-1.0）
    
    評価項目:
    1. 長さスコア（30%）
    2. 日本語含有率（40%）
    3. 句読点適切さ（20%）
    4. 語彙多様性（10%）
    """
    score = 0.0
    
    # 1. 長さスコア
    length = len(text)
    if 100 <= length <= 500:
        score += 0.3
    elif 500 < length <= 1000:
        score += 0.2
    elif 50 <= length < 100:
        score += 0.1
    
    # 2. 日本語含有率
    japanese_chars = count_japanese(text)
    japanese_ratio = japanese_chars / max(length, 1)
    score += japanese_ratio * 0.4
    
    # 3. 句読点
    punctuation = text.count('。') + text.count('、')
    expected_punct = length / 50
    if 0.5 * expected_punct <= punctuation <= 1.5 * expected_punct:
        score += 0.2
    
    # 4. 語彙多様性
    unique_words = len(set(text.split()))
    total_words = len(text.split())
    diversity = unique_words / max(total_words, 1)
    if diversity > 0.3:
        score += 0.1
    
    return min(score, 1.0)
```

### 重複除去アルゴリズム

```python
def remove_duplicates(samples: List[Dict]) -> List[Dict]:
    """
    MD5ハッシュベース重複除去
    """
    seen_hashes = set()
    unique_samples = []
    
    for sample in samples:
        text = sample.get('text') or sample.get('query', '')
        
        # 正規化
        normalized = text.lower().replace(' ', '').replace('\n', '')
        
        # ハッシュ計算
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_samples.append(sample)
    
    return unique_samples
```

## データセット統計

### 長さ分布

```
最小: 10文字
最大: 2048文字（トークン制限）
平均: 350文字
中央値: 280文字
標準偏差: 180文字

トークン換算（日本語）:
平均: ~500トークン
最大: ~3000トークン
```

### ドメイン分布（最終）

```
defense: 37,500 (12.5%)
aerospace: 37,500 (12.5%)
transport: 37,500 (12.5%)
medical: 37,500 (12.5%)
finance: 37,500 (12.5%)
business: 37,500 (12.5%)
information_system: 37,500 (12.5%)
general: 75,000 (25.0%)

合計: 300,000 (100%)
```

### 判定分布（合成データ）

```
ALLOW: 66,000 (33%)
ESCALATE: 68,000 (34%)
DENY: 66,000 (33%)

合計: 200,000 (100%)
```

## マルチモーダルデータ詳細

### 画像データ

#### カルテ画像
```
形式: JPEG, PNG
解像度: 1024x768以上
OCR対応: Tesseract日本語
前処理: ノイズ除去、CLAHE
```

#### 監視カメラ
```
形式: JPEG（連続フレーム）
解像度: 1920x1080（Full HD）
FPS: 5-10（リアルタイム監視）
検知: 顔認識、物体検出、差分検出
```

#### 文書画像
```
形式: PDF, JPEG, PNG
OCR: 日本語・英語対応
レイアウト解析: テーブル、図表抽出
```

### テキスト+画像統合

**データ構造**:
```json
{
  "id": "MM_MED_001",
  "domain": "medical",
  "modality": "multimodal",
  "text": {
    "content": "患者の症状について記録...",
    "length": 250
  },
  "image": {
    "path": "data/images/chart_001.jpg",
    "type": "medical_chart",
    "ocr_content": "血圧: 120/80, 体温: 36.5...",
    "size": "1024x768"
  },
  "safety_level": "極秘",
  "timestamp": "2025-11-06T12:00:00"
}
```

## データ前処理パイプライン

### ステップ1: テキスト正規化

```python
def normalize_text(text: str) -> str:
    # Unicode正規化
    text = unicodedata.normalize('NFKC', text)
    
    # 制御文字除去
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    
    # 連続空白削減
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
```

### ステップ2: トークナイズ

```python
def tokenize_sample(text: str, tokenizer, max_length: int = 2048):
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': encoding['input_ids']  # Causal LM
    }
```

### ステップ3: データ拡張（オプション）

```python
def augment_sample(text: str) -> List[str]:
    augmented = [text]  # オリジナル
    
    # 同義語置換
    augmented.append(synonym_replace(text))
    
    # 文順序シャッフル
    augmented.append(sentence_shuffle(text))
    
    # バックトランスレーション
    augmented.append(back_translate(text, 'en'))
    
    return augmented
```

## データセット分割戦略

### Train/Validation/Test

```
Total: 300,000 samples

Train: 270,000 (90%)
Validation: 20,000 (6.7%)
Test: 10,000 (3.3%)
```

### 分割方法

```python
def split_dataset(samples: List[Dict], 
                  train_ratio: float = 0.9,
                  val_ratio: float = 0.067,
                  seed: int = 42):
    random.seed(seed)
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]
    
    return train, val, test
```

### ドメイン層化抽出

```python
def stratified_split(samples: List[Dict], domains: List[str]):
    """
    各ドメインから同じ割合で分割
    """
    train, val, test = [], [], []
    
    for domain in domains:
        domain_samples = [s for s in samples if s['domain'] == domain]
        d_train, d_val, d_test = split_dataset(domain_samples)
        
        train.extend(d_train)
        val.extend(d_val)
        test.extend(d_test)
    
    return train, val, test
```

## データローディング最適化

### ストリーミング読み込み

```python
class StreamingDataset(IterableDataset):
    """
    メモリ効率的なストリーミングデータセット
    """
    
    def __init__(self, file_paths: List[Path], tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
    
    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get('text') or data.get('query', '')
                    
                    encoding = self.tokenizer(
                        text,
                        max_length=2048,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    
                    yield {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'labels': encoding['input_ids'].squeeze(0)
                    }
```

### DataLoader設定

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,          # 並列ロード
    pin_memory=True,        # GPU転送高速化
    prefetch_factor=2,      # プリフェッチ
    persistent_workers=True # ワーカー再利用
)
```

## 品質メトリクス統計

### 収集データ品質

```
平均品質スコア: 0.78
標準偏差: 0.12
最小値: 0.70（閾値）
最大値: 0.98

品質分布:
0.70-0.75: 25%
0.75-0.80: 35%
0.80-0.85: 25%
0.85+: 15%
```

### 合成データ品質

```
テンプレート適合率: 100%
三重推論統合率: 100%
identity_contract統合率: 100%
policy_state統合率: 100%

生成品質（人手評価サンプル100件）:
文法正確性: 95%
意味的整合性: 92%
ドメイン適合性: 90%
```

---

**付録C終了**

