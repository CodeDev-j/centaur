import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from langsmith import traceable

# Use standard postgres env vars (compatible with Azure/Docker)
DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "chiron_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_NAME = os.getenv("POSTGRES_DB", "lineage_db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

logger = logging.getLogger(__name__)
Base = declarative_base()

# --- ORM MODEL: THE LEDGER ---
class DocumentLedger(Base):
    """
    The 'State Truth'. Tracks every file processed by Chiron.
    """
    __tablename__ = "document_ledger"

    doc_hash = Column(String, primary_key=True, index=True) # SHA256 of file content
    filename = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    # Lineage logic
    is_superseded = Column(Boolean, default=False) # True if a newer draft exists
    superseded_by_hash = Column(String, nullable=True)
    
    # Audit logic
    processing_cost = Column(Float, default=0.0) # Tracking the "Latte Benchmark"
    status = Column(String, default="processing") # processing, completed, failed

class PostgresDriver:
    """
    Manages the connection to the Lineage Ledger.
    """
    def __init__(self):
        try:
            self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("✅ Connected to Postgres Lineage Ledger")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Postgres: {e}")
            raise

    @contextmanager
    @traceable(name="DB Session", run_type="tool")
    def session(self):
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"DB Session Error: {e}")
            raise
        finally:
            session.close()

# Singleton Instance
try:
    ledger_db = PostgresDriver()
except Exception:
    # Fallback for CI/CD environments where Docker might not be running yet
    ledger_db = None