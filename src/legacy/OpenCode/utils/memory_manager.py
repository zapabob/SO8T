"""
SO8T Memory Manager

This module provides SQLite-based memory management for the SO8T safety pipeline.
It handles conversation history, knowledge base, and SO(8) group state tracking.
"""

import sqlite3
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

class SO8TMemoryManager:
    """
    SO8T Memory Manager for conversation history and knowledge base management
    
    Features:
    - Conversation history storage and retrieval
    - Knowledge base management with embeddings
    - SO(8) group state tracking
    - Safety pattern management
    - Performance metrics logging
    - Automatic cleanup and optimization
    """
    
    def __init__(self, db_path: str = "database/so8t_memory.db"):
        """
        Initialize memory manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.conn = None
        self._connect()
        
        # Session management
        self.current_session_id = None
        
    def _connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=10000")
            
            logger.info(f"Connected to database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage in BLOB fields"""
        try:
            if isinstance(data, np.ndarray):
                return pickle.dumps(data)
            elif isinstance(data, (dict, list)):
                return pickle.dumps(data)
            else:
                return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            return b''
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from BLOB fields"""
        try:
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.current_session_id = session_id
        logger.info(f"Started session: {session_id}")
        return session_id
    
    def store_conversation(self, 
                          user_input: str,
                          safety_judgment: str,
                          model_response: str,
                          rotation_state: Optional[Any] = None,
                          confidence: float = 0.0,
                          processing_time_ms: int = 0,
                          input_type: str = 'text',
                          ocr_text: Optional[str] = None,
                          ocr_confidence: Optional[float] = None) -> int:
        """
        Store conversation entry
        
        Args:
            user_input: User input text
            safety_judgment: Safety judgment (ALLOW/ESCALATION/DENY)
            model_response: Model response text
            rotation_state: SO(8) group state
            confidence: Confidence score
            processing_time_ms: Processing time in milliseconds
            input_type: Input type (text/image/multimodal)
            ocr_text: OCR extracted text (for image inputs)
            ocr_confidence: OCR confidence score
            
        Returns:
            Conversation ID
        """
        try:
            if not self.current_session_id:
                self.start_session()
            
            # Serialize rotation state
            rotation_blob = self._serialize_data(rotation_state) if rotation_state else None
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO conversation_history 
                (session_id, user_input, safety_judgment, model_response, 
                 rotation_state, confidence_score, processing_time_ms, 
                 input_type, ocr_text, ocr_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                user_input,
                safety_judgment,
                model_response,
                rotation_blob,
                confidence,
                processing_time_ms,
                input_type,
                ocr_text,
                ocr_confidence
            ))
            
            conversation_id = cursor.lastrowid
            self.conn.commit()
            
            logger.debug(f"Stored conversation {conversation_id} in session {self.current_session_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return -1
    
    def get_conversation_history(self, session_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session ID (if None, uses current session)
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation dictionaries
        """
        try:
            if session_id is None:
                session_id = self.current_session_id
            
            if not session_id:
                return []
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, user_input, model_response, safety_judgment, confidence_score, 
                       rotation_state, processing_time_ms, input_type, 
                       ocr_text, ocr_confidence, timestamp
                FROM conversation_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (session_id, limit))
            
            conversations = []
            for row in cursor.fetchall():
                conversation = {
                    'id': row[0],
                    'user_input': row[1],
                    'model_response': row[2],
                    'safety_judgment': row[3],
                    'confidence': row[4],
                    'rotation_state': self._deserialize_data(row[5]) if row[5] else None,
                    'processing_time_ms': row[6],
                    'input_type': row[7],
                    'ocr_text': row[8],
                    'ocr_confidence': row[9],
                    'timestamp': row[10]
                }
                conversations.append(conversation)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def retrieve_context(self, session_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Retrieve conversation context
        
        Args:
            session_id: Session ID (uses current session if None)
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation dictionaries
        """
        try:
            if session_id is None:
                session_id = self.current_session_id
            
            if not session_id:
                logger.warning("No session ID provided")
                return []
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT user_input, model_response, safety_judgment, 
                       confidence_score, timestamp, input_type, ocr_text
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'user_input': row['user_input'],
                    'model_response': row['model_response'],
                    'safety_judgment': row['safety_judgment'],
                    'confidence_score': row['confidence_score'],
                    'timestamp': row['timestamp'],
                    'input_type': row['input_type'],
                    'ocr_text': row['ocr_text']
                })
            
            logger.debug(f"Retrieved {len(conversations)} conversations for session {session_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def store_knowledge(self, 
                       topic: str,
                       content: str,
                       embedding: Optional[np.ndarray] = None,
                       source_type: str = 'conversation',
                       source_id: Optional[int] = None,
                       confidence: float = 1.0) -> int:
        """
        Store knowledge in knowledge base
        
        Args:
            topic: Knowledge topic
            content: Knowledge content
            embedding: Vector embedding
            source_type: Source type (conversation/document/manual/distillation)
            source_id: Source ID reference
            confidence: Confidence score
            
        Returns:
            Knowledge ID
        """
        try:
            # Serialize embedding
            embedding_blob = self._serialize_data(embedding) if embedding is not None else None
            embedding_dim = embedding.shape[0] if embedding is not None else None
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO knowledge_base 
                (topic, content, embedding, embedding_dim, source_type, 
                 source_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                topic, content, embedding_blob, embedding_dim,
                source_type, source_id, confidence
            ))
            
            knowledge_id = cursor.lastrowid
            self.conn.commit()
            
            logger.debug(f"Stored knowledge {knowledge_id}: {topic}")
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            return -1
    
    def search_knowledge(self, 
                        query: str,
                        topic: Optional[str] = None,
                        min_confidence: float = 0.5,
                        limit: int = 10) -> List[Dict]:
        """
        Search knowledge base
        
        Args:
            query: Search query
            topic: Optional topic filter
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of knowledge entries
        """
        try:
            cursor = self.conn.cursor()
            
            if topic:
                cursor.execute("""
                    SELECT id, topic, content, confidence, created_at, last_accessed
                    FROM knowledge_base
                    WHERE topic = ? AND confidence >= ?
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT ?
                """, (topic, min_confidence, limit))
            else:
                cursor.execute("""
                    SELECT id, topic, content, confidence, created_at, last_accessed
                    FROM knowledge_base
                    WHERE (content LIKE ? OR topic LIKE ?) AND confidence >= ?
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT ?
                """, (f'%{query}%', f'%{query}%', min_confidence, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'topic': row['topic'],
                    'content': row['content'],
                    'confidence': row['confidence'],
                    'created_at': row['created_at'],
                    'last_accessed': row['last_accessed']
                })
            
            # Update access count
            if results:
                knowledge_ids = [r['id'] for r in results]
                placeholders = ','.join(['?' for _ in knowledge_ids])
                cursor.execute(f"""
                    UPDATE knowledge_base 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                """, knowledge_ids)
                self.conn.commit()
            
            logger.debug(f"Found {len(results)} knowledge entries for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    def store_so8_group_state(self, 
                             layer_index: int,
                             rotation_matrix: np.ndarray,
                             rotation_angles: np.ndarray,
                             group_stability: Optional[float] = None,
                             pet_penalty: Optional[float] = None) -> int:
        """
        Store SO(8) group state
        
        Args:
            layer_index: Layer index
            rotation_matrix: Rotation matrix
            rotation_angles: Rotation angles
            group_stability: Group stability metric
            pet_penalty: PET regularization penalty
            
        Returns:
            State ID
        """
        try:
            if not self.current_session_id:
                self.start_session()
            
            # Serialize rotation data
            rotation_matrix_blob = self._serialize_data(rotation_matrix)
            rotation_angles_blob = self._serialize_data(rotation_angles)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO so8_group_states 
                (session_id, layer_index, rotation_matrix, rotation_angles, 
                 group_stability, pet_penalty)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                layer_index,
                rotation_matrix_blob,
                rotation_angles_blob,
                group_stability,
                pet_penalty
            ))
            
            state_id = cursor.lastrowid
            self.conn.commit()
            
            logger.debug(f"Stored SO(8) group state {state_id} for layer {layer_index}")
            return state_id
            
        except Exception as e:
            logger.error(f"Error storing SO(8) group state: {e}")
            return -1
    
    def get_so8_group_states(self, session_id: Optional[str] = None, 
                            layer_index: Optional[int] = None) -> List[Dict]:
        """
        Retrieve SO(8) group states
        
        Args:
            session_id: Session ID (uses current session if None)
            layer_index: Optional layer index filter
            
        Returns:
            List of group state dictionaries
        """
        try:
            if session_id is None:
                session_id = self.current_session_id
            
            cursor = self.conn.cursor()
            
            if layer_index is not None:
                cursor.execute("""
                    SELECT id, layer_index, rotation_matrix, rotation_angles,
                           group_stability, pet_penalty, timestamp
                    FROM so8_group_states
                    WHERE session_id = ? AND layer_index = ?
                    ORDER BY timestamp DESC
                """, (session_id, layer_index))
            else:
                cursor.execute("""
                    SELECT id, layer_index, rotation_matrix, rotation_angles,
                           group_stability, pet_penalty, timestamp
                    FROM so8_group_states
                    WHERE session_id = ?
                    ORDER BY layer_index, timestamp DESC
                """, (session_id,))
            
            states = []
            for row in cursor.fetchall():
                states.append({
                    'id': row['id'],
                    'layer_index': row['layer_index'],
                    'rotation_matrix': self._deserialize_data(row['rotation_matrix']),
                    'rotation_angles': self._deserialize_data(row['rotation_angles']),
                    'group_stability': row['group_stability'],
                    'pet_penalty': row['pet_penalty'],
                    'timestamp': row['timestamp']
                })
            
            logger.debug(f"Retrieved {len(states)} SO(8) group states")
            return states
            
        except Exception as e:
            logger.error(f"Error retrieving SO(8) group states: {e}")
            return []
    
    def log_metric(self, 
                   metric_type: str,
                   metric_value: float,
                   threshold_value: Optional[float] = None,
                   status: str = 'pass',
                   details: Optional[str] = None) -> int:
        """
        Log performance metric
        
        Args:
            metric_type: Metric type
            metric_value: Metric value
            threshold_value: Threshold value
            status: Status (pass/fail/warning)
            details: Additional details
            
        Returns:
            Metric ID
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO model_metrics 
                (session_id, metric_type, metric_value, threshold_value, status, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                metric_type,
                metric_value,
                threshold_value,
                status,
                details
            ))
            
            metric_id = cursor.lastrowid
            self.conn.commit()
            
            logger.debug(f"Logged metric {metric_id}: {metric_type} = {metric_value}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error logging metric: {e}")
            return -1
    
    def get_session_statistics(self, session_id: Optional[str] = None) -> Dict:
        """
        Get session statistics
        
        Args:
            session_id: Session ID (uses current session if None)
            
        Returns:
            Session statistics dictionary
        """
        try:
            if session_id is None:
                session_id = self.current_session_id
            
            if not session_id:
                return {}
            
            cursor = self.conn.cursor()
            
            # Get conversation statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    AVG(confidence_score) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time,
                    COUNT(CASE WHEN safety_judgment = 'ALLOW' THEN 1 END) as allow_count,
                    COUNT(CASE WHEN safety_judgment = 'ESCALATION' THEN 1 END) as escalation_count,
                    COUNT(CASE WHEN safety_judgment = 'DENY' THEN 1 END) as deny_count
                FROM conversation_history
                WHERE session_id = ?
            """, (session_id,))
            
            conv_stats = cursor.fetchone()
            
            # Get knowledge statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_knowledge,
                    AVG(confidence) as avg_knowledge_confidence
                FROM knowledge_base
                WHERE source_id IN (
                    SELECT id FROM conversation_history WHERE session_id = ?
                )
            """, (session_id,))
            
            knowledge_stats = cursor.fetchone()
            
            return {
                'session_id': session_id,
                'total_conversations': conv_stats['total_conversations'] or 0,
                'avg_confidence': conv_stats['avg_confidence'] or 0.0,
                'avg_processing_time': conv_stats['avg_processing_time'] or 0.0,
                'allow_count': conv_stats['allow_count'] or 0,
                'escalation_count': conv_stats['escalation_count'] or 0,
                'deny_count': conv_stats['deny_count'] or 0,
                'total_knowledge': knowledge_stats['total_knowledge'] or 0,
                'avg_knowledge_confidence': knowledge_stats['avg_knowledge_confidence'] or 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """
        Clean up old data
        
        Args:
            days: Number of days to keep data
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor = self.conn.cursor()
            
            # Clean up old conversations
            cursor.execute("""
                DELETE FROM conversation_history 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            conv_deleted = cursor.rowcount
            
            # Clean up old SO(8) group states
            cursor.execute("""
                DELETE FROM so8_group_states 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            states_deleted = cursor.rowcount
            
            # Clean up old metrics
            cursor.execute("""
                DELETE FROM model_metrics 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            metrics_deleted = cursor.rowcount
            
            self.conn.commit()
            
            logger.info(f"Cleaned up old data: {conv_deleted} conversations, "
                       f"{states_deleted} group states, {metrics_deleted} metrics")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            cursor = self.conn.cursor()
            
            # Analyze tables for better query planning
            cursor.execute("ANALYZE")
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            
            self.conn.commit()
            logger.info("Database optimized")
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Test memory manager"""
    print("SO8T Memory Manager Test")
    print("=" * 50)
    
    # Create memory manager
    memory = SO8TMemoryManager("database/so8t_memory.db")
    
    try:
        # Start session
        session_id = memory.start_session()
        print(f"Started session: {session_id}")
        
        # Store test conversation
        conv_id = memory.store_conversation(
            user_input="こんにちは、元気ですか？",
            safety_judgment="ALLOW",
            model_response="こんにちは！元気です、ありがとうございます。",
            confidence=0.95,
            processing_time_ms=150
        )
        print(f"Stored conversation: {conv_id}")
        
        # Store knowledge
        knowledge_id = memory.store_knowledge(
            topic="greeting",
            content="Japanese greeting: こんにちは (konnichiwa)",
            confidence=0.9
        )
        print(f"Stored knowledge: {knowledge_id}")
        
        # Store SO(8) group state
        rotation_matrix = np.random.rand(8, 8)
        rotation_angles = np.random.rand(8)
        state_id = memory.store_so8_group_state(
            layer_index=0,
            rotation_matrix=rotation_matrix,
            rotation_angles=rotation_angles,
            group_stability=0.95,
            pet_penalty=0.01
        )
        print(f"Stored SO(8) group state: {state_id}")
        
        # Log metric
        metric_id = memory.log_metric(
            metric_type="safety_accuracy",
            metric_value=0.95,
            threshold_value=0.9,
            status="pass"
        )
        print(f"Logged metric: {metric_id}")
        
        # Retrieve context
        context = memory.retrieve_context(limit=5)
        print(f"Retrieved context: {len(context)} conversations")
        
        # Search knowledge
        knowledge_results = memory.search_knowledge("greeting", limit=5)
        print(f"Found knowledge: {len(knowledge_results)} entries")
        
        # Get session statistics
        stats = memory.get_session_statistics()
        print(f"Session statistics: {stats}")
        
        print("\n[OK] Memory manager test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return 1
    finally:
        memory.close()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
