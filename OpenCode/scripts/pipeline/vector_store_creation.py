#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG/CoG用ベクトルストアデータ作成モジュール

RAG（Retrieval-Augmented Generation）とCoG（Chain-of-Thought）用の
ベクトルストアデータを作成します。

Usage:
    python scripts/pipelines/vector_store_creation.py --input D:/webdataset/cleaned --output D:/webdataset/vector_stores
"""

import sys
import json
import logging
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter
import pickle

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vector_store_creation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGVectorStoreCreator:
    """RAG用ベクトルストアデータ作成クラス"""
    
    def __init__(self, output_dir: Path, chunk_size: int = 512, overlap: int = 128):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            chunk_size: チャンクサイズ（文字数）
            overlap: オーバーラップサイズ（文字数）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # RAG用ディレクトリ
        self.rag_dir = self.output_dir / 'rag_ready'
        self.rag_dir.mkdir(parents=True, exist_ok=True)
    
    def chunk_documents(self, samples: List[Dict]) -> List[Dict]:
        """
        ドキュメントをチャンクに分割
        
        Args:
            samples: サンプルリスト
        
        Returns:
            chunks: チャンクリスト
        """
        logger.info(f"[RAG] Chunking {len(samples)} documents...")
        
        chunks = []
        chunk_counter = 0
        
        for sample in samples:
            text = sample.get('cleaned_text', sample.get('text', ''))
            if not text or len(text.strip()) < 50:  # 最小長チェック
                continue
            
            # チャンク分割
            sample_chunks = self._split_text_into_chunks(text, sample)
            
            for chunk_text in sample_chunks:
                chunk_id = f"rag_chunk_{chunk_counter:08d}"
                chunk_counter += 1
                
                chunk = {
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_text,
                    'source_url': sample.get('url', ''),
                    'source_id': sample.get('id', ''),
                    'domain': sample.get('domain', 'unknown'),
                    'category': sample.get('category', 'unknown'),
                    'language': sample.get('language', 'unknown'),
                    'classification': sample.get('classification', {}),
                    'metadata': sample.get('metadata', {})
                }
                
                chunks.append(chunk)
        
        logger.info(f"[OK] Created {len(chunks)} chunks from {len(samples)} documents")
        return chunks
    
    def _split_text_into_chunks(self, text: str, sample: Dict) -> List[str]:
        """
        テキストをチャンクに分割
        
        Args:
            text: テキスト
            sample: サンプル情報
        
        Returns:
            chunks: チャンクリスト
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            
            # チャンクが空でない場合のみ追加
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # オーバーラップ分だけ進める
            start += self.chunk_size - self.overlap
            
            # 無限ループ防止
            if start >= text_length:
                break
        
        return chunks
    
    def save_rag_data(self, chunks: List[Dict]):
        """
        RAG用データを保存
        
        Args:
            chunks: チャンクリスト
        """
        logger.info(f"[RAG] Saving {len(chunks)} chunks to RAG-ready format...")
        
        # JSONL形式で保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.rag_dir / f"rag_chunks_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] RAG data saved to {output_file.name}")
        
        # メタデータインデックス作成
        self._create_metadata_index(chunks)
    
    def _create_metadata_index(self, chunks: List[Dict]):
        """
        メタデータインデックスを作成
        
        Args:
            chunks: チャンクリスト
        """
        logger.info("[RAG] Creating metadata index...")
        
        # 統計情報を収集
        by_language = Counter(chunk['language'] for chunk in chunks)
        by_domain = Counter(chunk['domain'] for chunk in chunks)
        by_category = Counter(chunk.get('category', 'unknown') for chunk in chunks)
        
        metadata = {
            'total_chunks': len(chunks),
            'created_at': datetime.now().isoformat(),
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'by_language': dict(by_language),
            'by_domain': dict(by_domain),
            'by_category': dict(by_category),
            'recommended_embedding_model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        }
        
        metadata_file = self.rag_dir / "rag_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[OK] Metadata index created: {metadata_file.name}")
        
        # RAG統合ガイド作成
        self._create_rag_integration_guide(metadata)
    
    def _create_rag_integration_guide(self, metadata: Dict):
        """
        RAG統合ガイドを作成
        
        Args:
            metadata: メタデータ
        """
        guide_content = f"""# RAG Integration Guide

## データセット情報
- **総チャンク数**: {metadata['total_chunks']:,}
- **作成日時**: {metadata['created_at']}
- **チャンクサイズ**: {metadata['chunk_size']} 文字
- **オーバーラップ**: {metadata['overlap']} 文字

## 言語分布
{self._format_dict(metadata['by_language'])}

## ドメイン分布
{self._format_dict(metadata['by_domain'])}

## カテゴリ分布
{self._format_dict(metadata['by_category'])}

## RAG統合手順

### Step 1: ベクトルDB準備

```python
# Chroma DB使用例
from chromadb import Client
from chromadb.config import Settings

client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="{self.rag_dir}/vector_db"
))

collection = client.create_collection("so8t_knowledge")
```

### Step 2: 埋め込み生成

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{metadata['recommended_embedding_model']}')

# チャンク読み込み&ベクトル化
import json

for file in Path('{self.rag_dir}').glob('rag_chunks_*.jsonl'):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            embedding = model.encode(chunk['chunk_text'])
            
            collection.add(
                ids=[chunk['chunk_id']],
                embeddings=[embedding],
                documents=[chunk['chunk_text']],
                metadatas=[{{
                    'url': chunk['source_url'],
                    'domain': chunk['domain'],
                    'language': chunk['language'],
                    'category': chunk.get('category', 'unknown')
                }}]
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
context = "\\n\\n".join(results['documents'][0])
prompt = f"Context:\\n{{context}}\\n\\nQuery: {{query}}"
```

## 推奨設定

- **Vector DB**: ChromaDB または FAISS
- **Embedding Model**: {metadata['recommended_embedding_model']}
- **Chunk size**: {metadata['chunk_size']}文字（調整可能）
- **Top-k**: 5-10チャンク
- **Re-ranking**: Optional（精度向上）

## 完了
データはRAG統合準備完了です。
"""
        
        guide_file = self.rag_dir / "RAG_INTEGRATION_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"[RAG] Integration guide created: {guide_file.name}")
    
    def _format_dict(self, d: Dict) -> str:
        """辞書フォーマット"""
        return '\n'.join([f"- **{k}**: {v:,}" for k, v in d.items()])


class CoGKnowledgeGraphCreator:
    """CoG（Chain-of-Thought）用ナレッジグラフ作成クラス"""
    
    def __init__(self, output_dir: Path):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CoG用ディレクトリ
        self.cog_dir = self.output_dir / 'cog_ready'
        self.cog_dir.mkdir(parents=True, exist_ok=True)
        
        # ナレッジグラフ用データ構造
        self.entities = {}  # {entity_id: entity_data}
        self.relations = []  # [{source, target, relation_type, metadata}]
        self.entity_counter = 0
    
    def extract_entities_and_relations(self, reasoning_text: str, layer: str, metadata: Dict) -> tuple:
        """
        推論テキストからエンティティとリレーションを抽出
        
        Args:
            reasoning_text: 推論テキスト
            layer: 推論レイヤー（task, safety, policy, final）
            metadata: メタデータ
        
        Returns:
            (entities, relations): エンティティとリレーションのリスト
        """
        entities = []
        relations = []
        
        # 簡易的なエンティティ抽出（実際の実装ではNERモデルを使用）
        # ここではキーワードベースの抽出を実装
        
        # エンティティ候補パターン（日本語・英語）
        entity_patterns = [
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # 英語の固有名詞
            r'[一-龠]+',  # 日本語の漢字
            r'[ァ-ヶー]+',  # カタカナ
            r'[ぁ-ん]+',  # ひらがな（一部の固有名詞）
        ]
        
        found_entities = set()
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, reasoning_text)
            for match in matches:
                if len(match) >= 2:  # 最小長チェック
                    found_entities.add(match.strip())
        
        # エンティティを作成
        for entity_text in found_entities:
            if entity_text not in self.entities:
                entity_id = f"entity_{self.entity_counter:08d}"
                self.entity_counter += 1
                
                entity = {
                    'entity_id': entity_id,
                    'name': entity_text,
                    'type': self._classify_entity_type(entity_text),
                    'layer': layer,
                    'metadata': metadata
                }
                
                self.entities[entity_text] = entity
                entities.append(entity)
        
        # リレーション抽出（簡易版：共起関係）
        entity_list = list(found_entities)
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                # 同じ文に出現するエンティティ間にリレーションを作成
                if self._entities_cooccur(entity1, entity2, reasoning_text):
                    relation = {
                        'source': self.entities[entity1]['entity_id'],
                        'target': self.entities[entity2]['entity_id'],
                        'relation_type': 'co_occurrence',
                        'layer': layer,
                        'confidence': 0.5,  # 簡易版のため固定値
                        'metadata': metadata
                    }
                    relations.append(relation)
        
        return entities, relations
    
    def _classify_entity_type(self, entity_text: str) -> str:
        """
        エンティティタイプを分類
        
        Args:
            entity_text: エンティティテキスト
        
        Returns:
            entity_type: エンティティタイプ
        """
        # 簡易的な分類（実際の実装ではNERモデルを使用）
        if re.match(r'^[A-Z][a-z]+$', entity_text):
            return 'PERSON'  # 人名（英語）
        elif re.match(r'^[一-龠]+$', entity_text):
            return 'ORGANIZATION'  # 組織名（漢字）
        elif re.match(r'^[ァ-ヶー]+$', entity_text):
            return 'LOCATION'  # 地名（カタカナ）
        else:
            return 'CONCEPT'  # 概念
    
    def _entities_cooccur(self, entity1: str, entity2: str, text: str) -> bool:
        """
        2つのエンティティが同じ文に出現するかチェック
        
        Args:
            entity1: エンティティ1
            entity2: エンティティ2
            text: テキスト
        
        Returns:
            True: 共起する、False: 共起しない
        """
        # 文分割（簡易版）
        sentences = re.split(r'[。！？\n]', text)
        
        for sentence in sentences:
            if entity1 in sentence and entity2 in sentence:
                return True
        
        return False
    
    def create_knowledge_graph(self, samples: List[Dict]) -> Dict:
        """
        ナレッジグラフを作成（SO8T四重推論からエンティティとリレーションを抽出）
        
        Args:
            samples: サンプルリスト（classificationフィールドにSO8T四重推論結果を含む）
        
        Returns:
            knowledge_graph: ナレッジグラフデータ
        """
        logger.info(f"[CoG] Creating knowledge graph from {len(samples)} samples...")
        
        # エンティティとリレーションをリセット
        self.entities = {}
        self.relations = []
        self.entity_counter = 0
        
        for sample in samples:
            classification = sample.get('classification', {})
            
            # SO8T四重推論結果を抽出
            task_reasoning = classification.get('task_reasoning', '')
            safety_reasoning = classification.get('safety_reasoning', '')
            policy_reasoning = classification.get('policy_reasoning', '')
            final_reasoning = classification.get('final_reasoning', '')
            
            # メタデータ
            sample_metadata = {
                'source_url': sample.get('url', ''),
                'source_id': sample.get('id', ''),
                'domain': sample.get('domain', 'unknown'),
                'category': sample.get('category', 'unknown'),
                'language': sample.get('language', 'unknown'),
                'confidence': classification.get('confidence', 0.0),
                'decision': classification.get('decision', '')
            }
            
            # 各推論レイヤーからエンティティとリレーションを抽出
            reasoning_layers = [
                ('task', task_reasoning),
                ('safety', safety_reasoning),
                ('policy', policy_reasoning),
                ('final', final_reasoning)
            ]
            
            for layer_name, reasoning_text in reasoning_layers:
                if not reasoning_text or len(reasoning_text.strip()) < 10:
                    continue
                
                entities, relations = self.extract_entities_and_relations(
                    reasoning_text, layer_name, sample_metadata
                )
                self.relations.extend(relations)
        
        knowledge_graph = {
            'entities': list(self.entities.values()),
            'relations': self.relations,
            'total_entities': len(self.entities),
            'total_relations': len(self.relations)
        }
        
        logger.info(f"[OK] Created knowledge graph: {len(self.entities)} entities, {len(self.relations)} relations")
        return knowledge_graph
    
    def save_knowledge_graph(self, knowledge_graph: Dict):
        """
        ナレッジグラフを保存（NetworkX形式、RDF形式、JSON形式）
        
        Args:
            knowledge_graph: ナレッジグラフデータ
        """
        logger.info(f"[CoG] Saving knowledge graph: {knowledge_graph['total_entities']} entities, {knowledge_graph['total_relations']} relations...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で保存
        json_file = self.cog_dir / f"knowledge_graph_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)
        logger.info(f"[OK] Knowledge graph JSON saved to {json_file.name}")
        
        # NetworkX形式で保存（オプション）
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            # エンティティをノードとして追加
            for entity in knowledge_graph['entities']:
                G.add_node(
                    entity['entity_id'],
                    name=entity['name'],
                    type=entity['type'],
                    layer=entity['layer']
                )
            
            # リレーションをエッジとして追加
            for relation in knowledge_graph['relations']:
                G.add_edge(
                    relation['source'],
                    relation['target'],
                    relation_type=relation['relation_type'],
                    layer=relation['layer'],
                    confidence=relation.get('confidence', 0.5)
                )
            
            # GraphML形式で保存
            graphml_file = self.cog_dir / f"knowledge_graph_{timestamp}.graphml"
            nx.write_graphml(G, str(graphml_file))
            logger.info(f"[OK] Knowledge graph GraphML saved to {graphml_file.name}")
            
        except ImportError:
            logger.warning("[CoG] NetworkX not available, skipping GraphML export")
        
        # RDF形式で保存（Turtle形式）
        rdf_file = self.cog_dir / f"knowledge_graph_{timestamp}.ttl"
        self._save_as_rdf(knowledge_graph, rdf_file)
        logger.info(f"[OK] Knowledge graph RDF saved to {rdf_file.name}")
        
        # メタデータインデックス作成
        self._create_metadata_index(knowledge_graph)
    
    def _save_as_rdf(self, knowledge_graph: Dict, output_file: Path):
        """
        RDF形式（Turtle）でナレッジグラフを保存
        
        Args:
            knowledge_graph: ナレッジグラフデータ
            output_file: 出力ファイルパス
        """
        rdf_lines = []
        rdf_lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        rdf_lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
        rdf_lines.append("@prefix so8t: <http://so8t.example.org/ontology#> .")
        rdf_lines.append("")
        
        # エンティティをRDF形式で記述
        for entity in knowledge_graph['entities']:
            entity_uri = f"so8t:{entity['entity_id']}"
            rdf_lines.append(f"{entity_uri} rdf:type so8t:Entity ;")
            rdf_lines.append(f"    rdfs:label \"{entity['name']}\" ;")
            rdf_lines.append(f"    so8t:entityType \"{entity['type']}\" ;")
            rdf_lines.append(f"    so8t:layer \"{entity['layer']}\" .")
            rdf_lines.append("")
        
        # リレーションをRDF形式で記述
        for relation in knowledge_graph['relations']:
            source_uri = f"so8t:{relation['source']}"
            target_uri = f"so8t:{relation['target']}"
            relation_type = relation['relation_type']
            
            rdf_lines.append(f"{source_uri} so8t:{relation_type} {target_uri} ;")
            rdf_lines.append(f"    so8t:layer \"{relation['layer']}\" ;")
            rdf_lines.append(f"    so8t:confidence {relation.get('confidence', 0.5)} .")
            rdf_lines.append("")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rdf_lines))
    
    def _create_metadata_index(self, knowledge_graph: Dict):
        """
        メタデータインデックスを作成
        
        Args:
            knowledge_graph: ナレッジグラフデータ
        """
        logger.info("[CoG] Creating metadata index...")
        
        # 統計情報を収集
        entities = knowledge_graph['entities']
        relations = knowledge_graph['relations']
        
        by_layer_entities = Counter(entity['layer'] for entity in entities)
        by_type_entities = Counter(entity['type'] for entity in entities)
        by_layer_relations = Counter(relation['layer'] for relation in relations)
        by_type_relations = Counter(relation['relation_type'] for relation in relations)
        
        metadata = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'created_at': datetime.now().isoformat(),
            'by_layer_entities': dict(by_layer_entities),
            'by_type_entities': dict(by_type_entities),
            'by_layer_relations': dict(by_layer_relations),
            'by_type_relations': dict(by_type_relations)
        }
        
        metadata_file = self.cog_dir / "knowledge_graph_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[OK] Metadata index created: {metadata_file.name}")
        
        # CoG統合ガイド作成
        self._create_cog_integration_guide(metadata)
    
    def _create_cog_integration_guide(self, metadata: Dict):
        """
        CoG統合ガイドを作成
        
        Args:
            metadata: メタデータ
        """
        # SPARQLクエリの例を別途定義（f-string内の?を避けるため）
        sparql_example = '''SELECT ?entity ?name ?type
    WHERE {
        ?entity rdfs:label ?name .
        ?entity so8t:entityType ?type .
        ?entity so8t:layer "task" .
    }'''
        
        guide_content = f"""# CoG (Chain-of-Thought) Knowledge Graph Integration Guide

## データセット情報
- **総エンティティ数**: {metadata['total_entities']:,}
- **総リレーション数**: {metadata['total_relations']:,}
- **作成日時**: {metadata['created_at']}

## エンティティレイヤー分布
{self._format_dict(metadata['by_layer_entities'])}

## エンティティタイプ分布
{self._format_dict(metadata['by_type_entities'])}

## リレーションレイヤー分布
{self._format_dict(metadata['by_layer_relations'])}

## リレーションタイプ分布
{self._format_dict(metadata['by_type_relations'])}

## ナレッジグラフ統合手順

### Step 1: NetworkX形式で読み込み

```python
import networkx as nx

# GraphML形式で読み込み
G = nx.read_graphml('{self.cog_dir}/knowledge_graph_*.graphml')

# エンティティ（ノード）を取得
entities = list(G.nodes(data=True))

# リレーション（エッジ）を取得
relations = list(G.edges(data=True))

# 特定のレイヤーのエンティティを検索
task_entities = [n for n, d in G.nodes(data=True) if d.get('layer') == 'task']
```

### Step 2: Neo4jにインポート

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# エンティティをノードとして作成
with driver.session() as session:
    for entity in entities:
        session.run(
            "CREATE (e:Entity {{entity_id: $id, name: $name, type: $type, layer: $layer}})",
            id=entity['entity_id'],
            name=entity['name'],
            type=entity['type'],
            layer=entity['layer']
        )
    
    # リレーションをエッジとして作成
    for relation in relations:
        session.run(
            "MATCH (a:Entity {{entity_id: $source}}), (b:Entity {{entity_id: $target}})"
            "CREATE (a)-[r:RELATES {{type: $type, layer: $layer, confidence: $confidence}}]->(b)",
            source=relation['source'],
            target=relation['target'],
            type=relation['relation_type'],
            layer=relation['layer'],
            confidence=relation.get('confidence', 0.5)
        )
```

### Step 3: RDF形式で読み込み

```python
from rdflib import Graph

# Turtle形式で読み込み
g = Graph()
g.parse('{self.cog_dir}/knowledge_graph_*.ttl', format='turtle')

# SPARQLクエリで検索
query = {sparql_example!r}

results = g.query(query)
for row in results:
    print(f"Entity: {{row.name}}, Type: {{row.type}}")
```

### Step 4: JSON形式で読み込み

```python
import json

# JSON形式で読み込み
with open('{self.cog_dir}/knowledge_graph_*.json', 'r', encoding='utf-8') as f:
    kg = json.load(f)

# エンティティを検索
task_entities = [e for e in kg['entities'] if e['layer'] == 'task']

# リレーションを検索
safety_relations = [r for r in kg['relations'] if r['layer'] == 'safety']
```

## 推奨設定

- **Graph Database**: Neo4j（本番環境推奨）または NetworkX（開発・テスト用）
- **RDF Format**: Turtle形式（標準的なRDF形式）
- **JSON Format**: 軽量な処理やAPI連携に適している
- **Layer Filtering**: レイヤー（task, safety, policy, final）でフィルタリング可能
- **Entity Type**: PERSON, ORGANIZATION, LOCATION, CONCEPT

## 完了
ナレッジグラフは統合準備完了です。
"""
        
        guide_file = self.cog_dir / "CoG_INTEGRATION_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"[CoG] Integration guide created: {guide_file.name}")
    
    def _format_dict(self, d: Dict) -> str:
        """辞書フォーマット"""
        return '\n'.join([f"- **{k}**: {v:,}" for k, v in d.items()])


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="RAG/CoG Vector Store Creation")
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory (cleaned data)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory (vector stores)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Chunk size for RAG (default: 512)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=128,
        help='Overlap size for RAG (default: 128)'
    )
    
    args = parser.parse_args()
    
    # 入力データを読み込み
    logger.info(f"[MAIN] Loading cleaned data from {args.input}...")
    samples = []
    
    # JSONLファイルを読み込み
    for jsonl_file in Path(args.input).glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"[OK] Loaded {len(samples)} samples")
    
    # RAG用ベクトルストア作成
    logger.info("[MAIN] Creating RAG vector store...")
    rag_creator = RAGVectorStoreCreator(
        output_dir=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    rag_chunks = rag_creator.chunk_documents(samples)
    rag_creator.save_rag_data(rag_chunks)
    
    # CoG用ナレッジグラフ作成
    logger.info("[MAIN] Creating CoG knowledge graph...")
    cog_creator = CoGKnowledgeGraphCreator(output_dir=args.output)
    knowledge_graph = cog_creator.create_knowledge_graph(samples)
    cog_creator.save_knowledge_graph(knowledge_graph)
    
    logger.info("[MAIN] Vector store creation completed successfully")


if __name__ == "__main__":
    main()

