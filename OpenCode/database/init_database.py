"""
SO8T Safety Pipeline Database Initialization Script

This script creates and initializes the SQLite database for the SO8T safety pipeline.
It sets up all tables, indexes, triggers, and initial data.
"""

import sqlite3
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SO8TDatabaseInitializer:
    """SO8T Safety Pipeline Database Initializer"""
    
    def __init__(self, db_path: str = "so8t_memory.db"):
        self.db_path = db_path
        self.schema_path = "create_schema.sql"
        
    def create_database(self):
        """Create and initialize the database"""
        try:
            # Check if database already exists
            if os.path.exists(self.db_path):
                logger.warning(f"Database {self.db_path} already exists. Backing up...")
                backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.db_path, backup_path)
                logger.info(f"Backup created: {backup_path}")
            
            # Create database connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Read and execute schema
            logger.info("Reading schema file...")
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Execute schema using executescript (handles multiple statements)
            logger.info("Creating database schema...")
            try:
                cursor.executescript(schema_sql)
                logger.info("Schema executed successfully")
            except sqlite3.Error as e:
                logger.error(f"Error executing schema: {e}")
                raise
            
            # Commit changes
            conn.commit()
            logger.info("Database schema created successfully")
            
            # Verify tables were created
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"Created tables: {[table[0] for table in tables]}")
            
            # Check initial data
            cursor.execute("SELECT COUNT(*) FROM safety_patterns;")
            pattern_count = cursor.fetchone()[0]
            logger.info(f"Inserted {pattern_count} safety patterns")
            
            # Create database info table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS database_info (
                    id INTEGER PRIMARY KEY,
                    version TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            
            cursor.execute("""
                INSERT INTO database_info (version, description) 
                VALUES (?, ?)
            """, ("1.0.0", "SO8T Safety Pipeline Database - Initial Version"))
            
            conn.commit()
            
            # Close connection
            conn.close()
            logger.info(f"Database initialized successfully: {self.db_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def verify_database(self):
        """Verify database integrity and structure"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check all required tables exist
            required_tables = [
                'conversation_history',
                'knowledge_base', 
                'safety_patterns',
                'model_metrics',
                'so8_group_states',
                'database_info'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [table[0] for table in cursor.fetchall()]
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = [idx[0] for idx in cursor.fetchall()]
            logger.info(f"Created {len(indexes)} indexes")
            
            # Check triggers
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
            triggers = [trg[0] for trg in cursor.fetchall()]
            logger.info(f"Created {len(triggers)} triggers")
            
            # Check views
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view';")
            views = [view[0] for view in cursor.fetchall()]
            logger.info(f"Created {len(views)} views")
            
            conn.close()
            logger.info("Database verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return False
    
    def get_database_info(self):
        """Get database information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get database info
            cursor.execute("SELECT * FROM database_info ORDER BY created_at DESC LIMIT 1;")
            db_info = cursor.fetchone()
            
            # Get table counts
            tables = ['conversation_history', 'knowledge_base', 'safety_patterns', 'model_metrics', 'so8_group_states']
            table_counts = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                table_counts[table] = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'database_info': db_info,
                'table_counts': table_counts,
                'file_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return None

def main():
    """Main function"""
    print("SO8T Safety Pipeline Database Initializer")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Initialize database
    initializer = SO8TDatabaseInitializer()
    
    print(f"Initializing database: {initializer.db_path}")
    if initializer.create_database():
        print("Database creation completed successfully!")
        
        # Verify database
        print("Verifying database...")
        if initializer.verify_database():
            print("Database verification passed!")
            
            # Show database info
            info = initializer.get_database_info()
            if info:
                print("\nDatabase Information:")
                print(f"  Version: {info['database_info'][1] if info['database_info'] else 'Unknown'}")
                print(f"  File Size: {info['file_size'] / 1024:.2f} KB")
                print(f"  Tables:")
                for table, count in info['table_counts'].items():
                    print(f"    {table}: {count} records")
        else:
            print("Database verification failed!")
            return 1
    else:
        print("Database creation failed!")
        return 1
    
    print("\nDatabase initialization completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
