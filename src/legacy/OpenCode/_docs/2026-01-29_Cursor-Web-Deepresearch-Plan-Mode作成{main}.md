# Cursor Web Deepresearch Plan Mode作成 実装ログ

## 実装情報
- **日付**: 2026-01-29
- **Worktree**: main
- **機能名**: Cursor Web Deepresearch Plan Mode作成（MCP browser tools統合）
- **実装者**: AI Agent

## 概要

CursorのMCP browser toolsを活用したDeepresearch + Plan Modeスキルを作成しました。WEB検索による深いリサーチ機能と、その結果を活用した計画作成・実行システムを統合しています。

## 実装内容

### 1. Web Deepresearch Plan Modeスキル作成

**ファイル**: `.cursor/skills/cursor-web-deepresearch-plan-mode/SKILL.md`

**主要機能**:
- **WEB検索統合**: CursorのMCP browser toolsを使用したリサーチ
- **深いリサーチ**: マルチソース検証と矛盾検出
- **計画生成**: リサーチ結果に基づく計画作成
- **実行統合**: リサーチを活用したステップ実行

### 2. コア機能

#### 2.1 WEB検索統合
- **browser_search**: 初期クエリ検索
- **browser_navigate**: ページへのナビゲーション
- **browser_snapshot**: ページコンテンツの分析
- **browser_tabs**: 複数タブの管理

#### 2.2 リサーチ方法論
- **マルチソース検証**: 3つ以上のソースからの相互参照
- **信頼性評価**: ソースの権威性と信頼性の評価
- **矛盾検出**: 矛盾する情報の識別と解決
- **タイムライン分析**: 情報の新しさと進化の追跡

#### 2.3 計画生成
- **リサーチ情報に基づく計画**: リサーチ結果に基づく計画作成
- **依存関係分析**: リサーチからステップ依存関係を識別
- **リスク評価**: リサーチ洞察に基づくリスク評価
- **ベストプラクティス統合**: リサーチされたベストプラクティスの組み込み

### 3. リサーチワークフロー

#### Step 1: 初期検索
```python
# browser_searchを使用して初期クエリ
search_query = "best practices for [task domain] implementation"
browser_search(query=search_query, max_results=10)
```

#### Step 2: コンテンツ分析
```python
# 有望な結果にナビゲート
tabs = browser_tabs(action="list")
for tab in tabs[:5]:
    browser_navigate(url=tab["url"])
    browser_lock()
    snapshot = browser_snapshot()
    analyze_content(snapshot)
    browser_unlock()
```

#### Step 3: マルチソース検証
```python
# 検証クエリで検索
validation_queries = [
    f"{topic} alternative approaches",
    f"{topic} comparison analysis"
]
```

#### Step 4: 情報統合
- アプローチの抽出
- ベストプラクティスの抽出
- 共通の落とし穴の抽出
- 推奨ツールの抽出
- 信頼性スコアの評価

### 4. 計画作成

#### リサーチ情報に基づく計画構造
```python
plan = {
    "id": "unique-plan-id",
    "title": task_description,
    "research_basis": research_summary,
    "steps": [],
    "sources": research_summary["sources"],
    "confidence_level": calculate_confidence(research_summary)
}
```

#### ステップ生成
- リサーチステップ（必要に応じて）
- ベストプラクティスからの実装ステップ
- 検証ステップ

### 5. Browser Tool使用パターン

#### パターン1: 順次リサーチ
- 検索 → ナビゲート → 分析 → 次へ

#### パターン2: 並列リサーチ
- 複数タブを開く → 並列分析

#### パターン3: 深いダイブ
- リンクをたどって深いリサーチ

### 6. リサーチ品質評価

#### ソース信頼性スコアリング
- ドメイン権威性
- コンテンツ品質
- 新しさ
- 専門性指標
- 引用数

#### 矛盾検出
- 主張の抽出
- 矛盾の識別
- 解決策の提案

### 7. 計画実行

#### ステップ固有リサーチ
- ステップ実行前にリサーチ
- リサーチコンテキストを使用した実行

#### 実行中のリサーチ検証
- 実装の検証情報を検索
- リサーチ基準に照らして検証

## ベストプラクティス

### リサーチ戦略
1. **広く始めて狭める**: 一般的なクエリから始め、焦点を絞る
2. **複数のソース**: 常に3つ以上の独立したソースで検証
3. **ソースの多様性**: 異なるタイプのソースを使用（docs、ブログ、フォーラム）
4. **新しさを確認**: 急速に変化するドメインでは最新情報を優先
5. **引用をたどる**: 可能な限り元のソースまで遡る

### Browser Tool使用
1. **タブをロック**: 操作前に常にタブをロック
2. **クリーンアップ**: 未使用タブを閉じてリソース管理
3. **スナップショットタイミング**: ページが完全に読み込まれた後にスナップショット
4. **ナビゲーション**: 効率的なブラウジングのためにbrowser_navigate_backを使用
5. **エラーハンドリング**: ナビゲーション失敗を適切に処理

### 計画生成
1. **まずリサーチ**: 計画前にリサーチを完了
2. **結果を統合**: 複数のソースからの洞察を組み合わせる
3. **ソースを文書化**: 計画にソースURLを含める
4. **動的に更新**: 新しいリサーチに基づいて計画を修正
5. **仮定を検証**: リサーチで計画の仮定を検証

## 使用例

### 完全なリサーチから計画へのフロー

```python
# 1. 初期リサーチ
task = "FastAPIでJWT認証を実装"

browser_search(query="JWT authentication FastAPI best practices 2026")
tabs = browser_tabs(action="list")

# 2. トップソースを分析
research_data = []
for tab in tabs[:5]:
    browser_navigate(url=tab["url"])
    browser_lock()
    snapshot = browser_snapshot()
    data = extract_research_data(snapshot)
    research_data.append(data)
    browser_unlock()

# 3. 代替ソースで検証
browser_search(query="JWT FastAPI security considerations")

# 4. リサーチを統合
research_summary = synthesize_research(research_data)

# 5. 計画作成
plan = create_research_plan(task, research_summary)

# 6. トラッキング初期化
todo_write(merge=False, todos=plan["steps"])

# 7. リサーチ統合で実行
execute_plan_with_research(plan)
```

## エラーハンドリング

### Browser Navigationエラー
- 最大3回のリトライ
- 指数バックオフ
- エラーログ記録

### リサーチ失敗処理
- プライマリソースの失敗時はフォールバックソースを使用
- 部分的な結果の許容
- エラーの文書化

## パフォーマンス最適化

### リサーチ結果のキャッシュ
- クエリハッシュによるキャッシュキー
- タイムスタンプによる古さチェック
- キャッシュヒット率の最適化

### 並列リサーチ
- 複数クエリの並列実行
- 非同期処理による効率化
- 結果の統合

## Cursor機能との統合

### Todo管理
- `todo_write`で計画ステップを追跡
- ステップ説明にリサーチステータスを含める
- リサーチステップの完了をマーク

### ファイル管理
- リサーチ結果をファイルに保存
- 計画ファイルにソースURLを文書化
- リサーチサマリードキュメントの作成

### ターミナル統合
- リサーチ結果に基づいてコマンドを実行
- リサーチに基づくチェックで実装を検証
- リサーチに基づいてテストを実行

## リサーチ文書化

### リサーチログ形式
```markdown
# Research Log: [Task Name]

## Research Query
[Initial search query]

## Sources Analyzed
1. [Source 1 URL]
   - Credibility: High
   - Key Findings: [Summary]

## Synthesis
[Combined insights from all sources]

## Plan Implications
[How research informs the plan]
```

## 次のステップ

### 実装予定機能
1. **自動リサーチ統合**: ステップ実行時の自動リサーチ
2. **リサーチ結果の可視化**: リサーチ結果のグラフィカル表示
3. **リサーチテンプレート**: 一般的なタスクのリサーチテンプレート
4. **コラボレーション機能**: 複数エージェント間のリサーチ共有

### 最適化予定
1. **リサーチキャッシュの改善**: より効率的なキャッシュ戦略
2. **並列リサーチの最適化**: より高速な並列処理
3. **リサーチ品質スコアリング**: より正確な信頼性評価

## 関連ファイル

- `.cursor/skills/cursor-web-deepresearch-plan-mode/SKILL.md`: Web Deepresearch Plan Modeスキル
- `.cursor/skills/cursor-claudecode-plan-mode/SKILL.md`: ClaudeCode Plan Modeスキル
- `C:\Users\downl\.cursor\skills\web-search-deepresearch\SKILL.md`: web-search-deepresearchスキル

---

*実装完了日時: 2026-01-29*
*SO8T Research Initiative*
