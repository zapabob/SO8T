# 2026-02-09_GitIgnoreFix_NulDevice.md

## 概要

Windows環境における Git の `error: short read while indexing nul` エラーを解消した。

## 原因分析

ルートディレクトリに `nul` という名前の物理ファイルが存在していた。Windowsにおいて `NUL` は予約済みのデバイス名であり、Git がこのファイルをインデックス（ステージング）しようとした際、デバイスとして扱われ読み取りに失敗していた。

## 修正内容

### 1. 物理ファイルの削除

- Windowsの予約名を扱うための特殊パス `\\.\` を使用して、ルートの `nul` ファイルを強制削除した。
  - コマンド: `cmd /c "del \\.\c:\Users\downl\Desktop\SO8T\nul"`

### 2. .gitignore の更新

- 今後同様の問題が発生しないよう、Windowsの予約デバイス名を `.gitignore` に追加した。
- 追加内容:
  ```gitignore
  # Windows Reserved Device Names
  CON
  PRN
  AUX
  NUL
  COM[1-9]
  LPT[1-9]
  nul
  ```

## 検証結果

- `git add -A` コマンドがエラーなく実行できるようになった。
- `ls nul` でファイルが存在しないことを確認。
