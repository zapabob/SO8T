# 2026-02-05 Git リポジトリエラー修正ログ (Submodule Flattening)

## 概要

`git status` 実行時に発生していた `fatal: '.git' not recognized as a git repository` エラーを解消しました。

## 原因分析

リポジトリ内の以下のディレクトリが Git インデックス上でサブモジュール（mode 160000 / gitlink）として記録されていましたが、内部の `.git` ディレクトリが不完全、あるいは破損した状態で存在していたため、メインリポジトリの Git がこれらを正しく解釈できずエラーとなっていました。

**対象ディレクトリ:**

- `datasets/nsfw_base_model`
- `datasets/nsfw_dataset`
- `datasets/nsfw_training_data`
- `computer-use-preview`

## 実施した対策

不完全なサブモジュール構造を解消し、通常のディレクトリとしてリポジトリに統合（Flattening）しました。

1. **インデックスからの削除**: `git rm --cached <path>` により、サブモジュールとしての登録を解除。
2. **破損エントリの除去**: ディレクトリ内の不完全な `.git` フォルダを削除。
3. **正規ディレクトリ化**: `git add <path>` により、内部のファイルをメインリポジトリの管理対象として再登録。

## 検証結果

- `git status -uall` が正常に実行され、エラーが解消されたことを確認済み。
- リポジトリの再編が完了し、正常な開発・同期作業が可能な状態に復旧しました。
