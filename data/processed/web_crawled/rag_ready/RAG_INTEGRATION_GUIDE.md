# RAG Integration Guide

## データセット情報
- **総チャンク数**: 0
- **作成日時**: 2025-11-08T01:41:23.918603
- **チャンクサイズ**: 512 文字
- **オーバーラップ**: 128 文字

## 言語分布


## ドメイン分布


## RAG統合手順

### Step 1: ベクトルDB準備

```python
# Chroma DB使用例
from chromadb import Client
from chromadb.config import Settings

client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="data\processed\web_crawled\rag_ready/vector_db"
))

collection = client.create_collection("so8t_knowledge")
```

### Step 2: 埋め込み生成

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# チャンク読み込み&ベクトル化
import json

for file in Path('data\processed\web_crawled\rag_ready').glob('rag_chunks_*.jsonl'):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            embedding = model.encode(chunk['chunk_text'])
            
            collection.add(
                ids=[chunk['chunk_id']],
                embeddings=[embedding],
                documents=[chunk['chunk_text']],
                metadatas=[{
                    'url': chunk['source_url'],
                    'domain': chunk['domain'],
                    'language': chunk['language']
                }]
            )
```

### Step 3: RAG検索

```python
# クエリ検索
query = "防衛装備品の調達について"
query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# 結果を SO8T モデルに渡す
context = "\n\n".join(results['documents'][0])
prompt = f"Context:\n{context}\n\nQuery: {query}"
```

## 推奨設定

- **Vector DB**: ChromaDB または FAISS
- **Embedding Model**: paraphrase-multilingual-mpnet-base-v2
- **Chunk size**: 512文字（調整可能）
- **Top-k**: 5-10チャンク
- **Re-ranking**: Optional（精度向上）

## 完了
データはRAG統合準備完了です。
