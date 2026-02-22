import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, Date, Text, JSON, ForeignKey, text
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

class DocumentMeta(Base):
    """
    Semantic metadata for ingested documents.
    Separate from DocumentLedger (ingestion state) for clean separation of concerns.
    Auto-populated at ingestion; user-editable via API.
    """
    __tablename__ = "document_meta"

    doc_hash = Column(String, ForeignKey("document_ledger.doc_hash", ondelete="CASCADE"),
                      primary_key=True, index=True)

    # ── Identity ──────────────────────────────────────────
    company_name = Column(String, nullable=True)
    document_type = Column(String, nullable=True)
    project_code = Column(String, nullable=True)

    # ── Temporal ──────────────────────────────────────────
    as_of_date = Column(Date, nullable=True)
    period_label = Column(String, nullable=True)
    publish_date = Column(Date, nullable=True)

    # ── Classification ────────────────────────────────────
    sector = Column(String, nullable=True)
    geography = Column(String, nullable=True)
    currency = Column(String, nullable=True)
    confidentiality = Column(String, nullable=True)
    language = Column(String, server_default="en")

    # ── Physical ──────────────────────────────────────────
    page_count = Column(Integer, server_default="0")

    # ── User-managed ──────────────────────────────────────
    tags = Column(JSON, server_default="[]")
    notes = Column(Text, nullable=True)

    # ── Provenance ────────────────────────────────────────
    extraction_confidence = Column(Float, server_default="0.0")
    user_overrides = Column(JSON, server_default="{}")
    last_edited_at = Column(DateTime, nullable=True)


def _meta_to_dict(meta: DocumentMeta) -> dict:
    """Convert a DocumentMeta ORM instance to a plain dict."""
    return {
        "doc_hash": meta.doc_hash,
        "company_name": meta.company_name,
        "document_type": meta.document_type,
        "project_code": meta.project_code,
        "as_of_date": meta.as_of_date.isoformat() if meta.as_of_date else None,
        "period_label": meta.period_label,
        "publish_date": meta.publish_date.isoformat() if meta.publish_date else None,
        "sector": meta.sector,
        "geography": meta.geography,
        "currency": meta.currency,
        "confidentiality": meta.confidentiality,
        "language": meta.language,
        "page_count": meta.page_count,
        "tags": meta.tags or [],
        "notes": meta.notes,
        "extraction_confidence": meta.extraction_confidence,
        "user_overrides": meta.user_overrides or {},
        "last_edited_at": meta.last_edited_at.isoformat() if meta.last_edited_at else None,
    }


class PostgresDriver:
    """
    Manages the connection to the Lineage Ledger.
    """
    def __init__(self):
        try:
            self.engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
            )
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

    # ── Document Metadata CRUD ────────────────────────────────────────

    def upsert_document_meta(self, doc_hash: str, **fields) -> dict:
        """Create or update document metadata. Returns the full row as dict."""
        with self.session() as s:
            meta = s.query(DocumentMeta).filter_by(doc_hash=doc_hash).first()
            if not meta:
                meta = DocumentMeta(doc_hash=doc_hash, **fields)
                s.add(meta)
            else:
                overrides = meta.user_overrides or {}
                for k, v in fields.items():
                    if k not in overrides:
                        setattr(meta, k, v)
            s.flush()
            return _meta_to_dict(meta)

    def get_document_meta(self, doc_hash: str) -> dict | None:
        """Get full metadata for a document."""
        with self.session() as s:
            meta = s.query(DocumentMeta).filter_by(doc_hash=doc_hash).first()
            return _meta_to_dict(meta) if meta else None

    def update_document_meta(self, doc_hash: str, **fields) -> dict | None:
        """User-initiated update. Marks fields as user-overridden."""
        with self.session() as s:
            meta = s.query(DocumentMeta).filter_by(doc_hash=doc_hash).first()
            if not meta:
                return None
            overrides = dict(meta.user_overrides or {})
            for k, v in fields.items():
                if hasattr(meta, k) and k not in ("doc_hash", "extraction_confidence", "user_overrides"):
                    setattr(meta, k, v)
                    overrides[k] = True
            meta.user_overrides = overrides
            meta.last_edited_at = datetime.utcnow()
            s.flush()
            return _meta_to_dict(meta)

    def get_document_facets(self) -> dict:
        """Distinct values for filter dropdowns."""
        with self.session() as s:
            companies = [r[0] for r in s.query(DocumentMeta.company_name).distinct()
                         if r[0] is not None]
            sectors = [r[0] for r in s.query(DocumentMeta.sector).distinct()
                       if r[0] is not None]
            types = [r[0] for r in s.query(DocumentMeta.document_type).distinct()
                     if r[0] is not None]
            projects = [r[0] for r in s.query(DocumentMeta.project_code).distinct()
                        if r[0] is not None]
            return {
                "companies": sorted(companies),
                "sectors": sorted(sectors),
                "document_types": sorted(types),
                "projects": sorted(projects),
            }


# Singleton Instance
try:
    ledger_db = PostgresDriver()
except Exception:
    # Fallback for CI/CD environments where Docker might not be running yet
    ledger_db = None