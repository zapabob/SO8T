# 合成データ生成レポート

## 生成概要
- **生成日時**: 2025-11-06T23:03:07.955705
- **総サンプル数**: 100,000
- **ドメイン数**: 4

## ドメイン別統計
- **defense**: 25,000 samples (25.0%)
- **aerospace**: 25,000 samples (25.0%)
- **transport**: 25,000 samples (25.0%)
- **general**: 25,000 samples (25.0%)

## 判定分布
- **DENY**: 32,826 samples (32.8%)
- **ESCALATE**: 34,103 samples (34.1%)
- **ALLOW**: 33,071 samples (33.1%)

## データ構造
各サンプルには以下の情報が含まれます：
- クエリ（query）
- 応答（response）
- 判定（decision: ALLOW/ESCALATE/DENY）
- 推論根拠（reasoning）
- リスクレベル（risk_level）
- ポリシー参照（policy_ref）
- 役割契約（identity_contract）
- ポリシー状態（policy_state）
- エスカレーション先（escalation_target）
- メタデータ（metadata）

## ステータス
- [OK] 合成データ生成完了
- [OK] 三重推論データ統合
- [OK] identity_contract統合
- [OK] policy_state統合
- [OK] エスカレーション情報統合
