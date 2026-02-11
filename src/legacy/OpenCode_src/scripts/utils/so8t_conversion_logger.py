"""
SO8T GGUF Conversion Logger

This module provides SQLite-based logging for SO8T GGUF conversion processes.
It tracks conversion sessions, layer transformations, and metadata for debugging
and auditing purposes.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid

logger = logging.getLogger(__name__)


class SO8TConversionLogger:
    """
    SO8T GGUF Conversion Logger
    
    Features:
    - Conversion session tracking
    - Layer-by-layer conversion logging
    - Metadata storage
    - Performance metrics
    - Error tracking
    """
    
    def __init__(self, db_path: str = "database/so8t_conversion.db"):
        """
        Initialize conversion logger
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.conn = None
        self.current_session_id = None
        self._connect()
        self._create_tables()
        
    def _connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=10000")
            
            logger.info(f"Connected to conversion logger database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Conversion sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversion_sessions (
                    session_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    output_path TEXT,
                    total_layers INTEGER,
                    conversion_status TEXT DEFAULT 'in_progress',
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds REAL,
                    error_message TEXT,
                    so8t_version TEXT,
                    python_version TEXT,
                    torch_version TEXT
                )
            """)
            
            # Layer conversions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS layer_conversions (
                    conversion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    layer_index INTEGER,
                    layer_name TEXT NOT NULL,
                    tensor_name TEXT NOT NULL,
                    original_shape TEXT NOT NULL,
                    converted_shape TEXT,
                    original_dtype TEXT NOT NULL,
                    target_dtype TEXT,
                    quant_type TEXT,
                    conversion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_time_ms REAL,
                    memory_usage_mb REAL,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    FOREIGN KEY (session_id) REFERENCES conversion_sessions(session_id)
                )
            """)
            
            # Metadata entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata_entries (
                    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    metadata_key TEXT NOT NULL,
                    metadata_value TEXT,
                    metadata_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversion_sessions(session_id)
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversion_sessions(session_id)
                )
            """)
            
            # SO8T specific parameters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS so8t_parameters (
                    param_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    rotation_dim INTEGER DEFAULT 8,
                    pet_lambda REAL DEFAULT 0.01,
                    safety_weight REAL DEFAULT 0.1,
                    cmd_weight REAL DEFAULT 0.9,
                    triality_enabled BOOLEAN DEFAULT 1,
                    multimodal_enabled BOOLEAN DEFAULT 1,
                    ocr_enabled BOOLEAN DEFAULT 1,
                    safety_classes TEXT DEFAULT 'ALLOW,ESCALATION,DENY',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversion_sessions(session_id)
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_layer_conversions_session 
                ON layer_conversions(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_entries_session 
                ON metadata_entries(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_session 
                ON performance_metrics(session_id)
            """)
            
            self.conn.commit()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def start_conversion(
        self,
        model_name: str,
        model_type: str,
        source_path: str,
        output_path: Optional[str] = None,
        total_layers: Optional[int] = None,
        so8t_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new conversion session
        
        Args:
            model_name: Name of the model being converted
            model_type: Type of model (e.g., "SO8TTransformer")
            source_path: Path to source model file
            output_path: Path to output GGUF file
            total_layers: Total number of layers to convert
            so8t_params: SO8T-specific parameters
            
        Returns:
            Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            self.current_session_id = session_id
            
            # Get version information
            import sys
            import torch
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            torch_version = torch.__version__
            so8t_version = "1.0.0"
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO conversion_sessions (
                    session_id, model_name, model_type, source_path, output_path,
                    total_layers, so8t_version, python_version, torch_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, model_name, model_type, source_path, output_path,
                total_layers, so8t_version, python_version, torch_version
            ))
            
            # Log SO8T parameters if provided
            if so8t_params:
                cursor.execute("""
                    INSERT INTO so8t_parameters (
                        session_id, rotation_dim, pet_lambda, safety_weight, cmd_weight,
                        triality_enabled, multimodal_enabled, ocr_enabled, safety_classes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    so8t_params.get('rotation_dim', 8),
                    so8t_params.get('pet_lambda', 0.01),
                    so8t_params.get('safety_weight', 0.1),
                    so8t_params.get('cmd_weight', 0.9),
                    so8t_params.get('triality_enabled', True),
                    so8t_params.get('multimodal_enabled', True),
                    so8t_params.get('ocr_enabled', True),
                    so8t_params.get('safety_classes', 'ALLOW,ESCALATION,DENY')
                ))
            
            self.conn.commit()
            logger.info(f"Started conversion session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start conversion: {e}")
            raise
    
    def log_layer_conversion(
        self,
        layer_index: Optional[int],
        layer_name: str,
        tensor_name: str,
        original_shape: tuple,
        original_dtype: str,
        converted_shape: Optional[tuple] = None,
        target_dtype: Optional[str] = None,
        quant_type: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Log a layer conversion
        
        Args:
            layer_index: Index of the layer being converted
            layer_name: Name of the layer
            tensor_name: Name of the tensor
            original_shape: Original tensor shape
            original_dtype: Original data type
            converted_shape: Converted tensor shape
            target_dtype: Target data type
            quant_type: Quantization type applied
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in megabytes
            success: Whether conversion was successful
            error_message: Error message if failed
        """
        try:
            if self.current_session_id is None:
                logger.warning("No active conversion session")
                return
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO layer_conversions (
                    session_id, layer_index, layer_name, tensor_name,
                    original_shape, converted_shape, original_dtype, target_dtype,
                    quant_type, processing_time_ms, memory_usage_mb,
                    success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id, layer_index, layer_name, tensor_name,
                str(original_shape), str(converted_shape) if converted_shape else None,
                original_dtype, target_dtype, quant_type, processing_time_ms,
                memory_usage_mb, success, error_message
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log layer conversion: {e}")
    
    def log_metadata(
        self,
        key: str,
        value: Any,
        value_type: Optional[str] = None
    ):
        """
        Log metadata entry
        
        Args:
            key: Metadata key
            value: Metadata value
            value_type: Type of metadata value
        """
        try:
            if self.current_session_id is None:
                logger.warning("No active conversion session")
                return
            
            # Convert value to string
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
                value_type = value_type or 'json'
            else:
                value_str = str(value)
                value_type = value_type or type(value).__name__
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO metadata_entries (
                    session_id, metadata_key, metadata_value, metadata_type
                ) VALUES (?, ?, ?, ?)
            """, (self.current_session_id, key, value_str, value_type))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log metadata: {e}")
    
    def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None
    ):
        """
        Log performance metric
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_unit: Unit of measurement
        """
        try:
            if self.current_session_id is None:
                logger.warning("No active conversion session")
                return
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (
                    session_id, metric_name, metric_value, metric_unit
                ) VALUES (?, ?, ?, ?)
            """, (self.current_session_id, metric_name, metric_value, metric_unit))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log performance metric: {e}")
    
    def end_conversion(
        self,
        status: str = 'completed',
        error_message: Optional[str] = None
    ):
        """
        End the current conversion session
        
        Args:
            status: Final status ('completed', 'failed', 'cancelled')
            error_message: Error message if failed
        """
        try:
            if self.current_session_id is None:
                logger.warning("No active conversion session")
                return
            
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE conversion_sessions
                SET conversion_status = ?,
                    error_message = ?,
                    end_time = CURRENT_TIMESTAMP,
                    duration_seconds = (
                        julianday(CURRENT_TIMESTAMP) - julianday(start_time)
                    ) * 86400
                WHERE session_id = ?
            """, (status, error_message, self.current_session_id))
            
            self.conn.commit()
            logger.info(f"Ended conversion session: {self.current_session_id} (status: {status})")
            
            self.current_session_id = None
            
        except Exception as e:
            logger.error(f"Failed to end conversion: {e}")
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of conversion session
        
        Args:
            session_id: Session ID (uses current session if None)
            
        Returns:
            Dictionary containing session summary
        """
        try:
            sid = session_id or self.current_session_id
            if sid is None:
                return {}
            
            cursor = self.conn.cursor()
            
            # Get session info
            cursor.execute("""
                SELECT * FROM conversion_sessions WHERE session_id = ?
            """, (sid,))
            session_row = cursor.fetchone()
            
            if not session_row:
                return {}
            
            # Get layer conversion stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_conversions,
                    AVG(processing_time_ms) as avg_processing_time_ms,
                    SUM(memory_usage_mb) as total_memory_usage_mb
                FROM layer_conversions
                WHERE session_id = ?
            """, (sid,))
            stats_row = cursor.fetchone()
            
            # Get SO8T parameters
            cursor.execute("""
                SELECT * FROM so8t_parameters WHERE session_id = ?
            """, (sid,))
            params_row = cursor.fetchone()
            
            summary = {
                'session_id': sid,
                'model_name': session_row['model_name'],
                'model_type': session_row['model_type'],
                'status': session_row['conversion_status'],
                'duration_seconds': session_row['duration_seconds'],
                'total_layers': session_row['total_layers'],
                'stats': {
                    'total_conversions': stats_row['total_conversions'],
                    'successful_conversions': stats_row['successful_conversions'],
                    'avg_processing_time_ms': stats_row['avg_processing_time_ms'],
                    'total_memory_usage_mb': stats_row['total_memory_usage_mb']
                } if stats_row else {},
                'so8t_params': dict(params_row) if params_row else {}
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Closed conversion logger database connection")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Create logger
    conv_logger = SO8TConversionLogger()
    
    # Start conversion
    session_id = conv_logger.start_conversion(
        model_name="SO8T-Distilled-Safety",
        model_type="SO8TTransformer",
        source_path="models/so8t_distilled_safety.pt",
        output_path="models/so8t_distilled_safety.gguf",
        total_layers=28,
        so8t_params={
            'rotation_dim': 8,
            'pet_lambda': 0.01,
            'triality_enabled': True,
            'multimodal_enabled': True
        }
    )
    
    # Log layer conversion
    conv_logger.log_layer_conversion(
        layer_index=0,
        layer_name="embed_tokens",
        tensor_name="model.embed_tokens.weight",
        original_shape=(152064, 3584),
        original_dtype="float32",
        converted_shape=(152064, 3584),
        target_dtype="float16",
        quant_type="F16",
        processing_time_ms=125.5,
        memory_usage_mb=2048.0,
        success=True
    )
    
    # Log metadata
    conv_logger.log_metadata("so8t.group_structure", "SO8")
    conv_logger.log_metadata("so8t.triality_enabled", True)
    
    # Log performance metrics
    conv_logger.log_performance_metric("conversion_speed_mb_per_sec", 32.5, "MB/s")
    
    # End conversion
    conv_logger.end_conversion(status='completed')
    
    # Get summary
    summary = conv_logger.get_session_summary(session_id)
    print("\nConversion Summary:")
    print(json.dumps(summary, indent=2))
    
    # Close
    conv_logger.close()

