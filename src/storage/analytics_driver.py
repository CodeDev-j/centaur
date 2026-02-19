"""
Analytics Driver: Structured Metric Store (Postgres).

Flattens MetricSeries into one row per DataPoint in a `metric_facts` table.
Enables SQL-based structured queries over financial data (e.g., "revenue > $1B").

Key design decisions:
- `resolved_value` is PRE-COMPUTED in Python (never by the LLM in Text-to-SQL).
- `period_date` is PRE-PARSED from labels for temporal filtering.
- Uses existing Postgres instance (same as document_ledger).
"""

import logging
import re
from datetime import date, datetime
from typing import List, Dict, Optional, Any
from uuid import uuid4

from sqlalchemy import (
    create_engine, Column, String, Float, Date, DateTime, Text,
    text as sql_text,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from langsmith import traceable

from src.config import SystemConfig
from src.schemas.deal_stream import UnifiedDocument, VisualItem, ChartTableItem
from src.schemas.vision_output import MetricSeries, DataPoint

logger = logging.getLogger(__name__)

Base = declarative_base()

# ==============================================================================
# MAGNITUDE RESOLUTION
# ==============================================================================

MAGNITUDE_MAP = {
    "k": 1e3,
    "M": 1e6,
    "B": 1e9,
    "T": 1e12,
    "None": 1.0,
}


def _resolve_value(numeric_value: Optional[float], magnitude: str) -> Optional[float]:
    """
    Deterministic magnitude resolution. Never relies on LLM math.
    Example: numeric_value=12.4, magnitude='B' → 12_400_000_000.0
    """
    if numeric_value is None:
        return None
    multiplier = MAGNITUDE_MAP.get(magnitude, 1.0)
    return numeric_value * multiplier


def _parse_period_date(label: str) -> Optional[date]:
    """
    Best-effort date parsing from axis labels.
    Handles: '2024', 'FY2023', 'Q1 2024', 'Q3'24', 'H1 2023', '9M 2024',
    'LTM Sep-23', '2024E', '2025F', 'Jan-24', 'Mar 2024'.
    Returns None if unparsable (not an error — some labels are non-temporal).
    """
    clean = label.strip()

    # Strip estimate/forecast suffixes: '2024E', '2025F', '2023A', '2024P'
    clean = re.sub(r'[AEFP]$', '', clean)

    # "FY2024" or "FY 2024"
    m = re.match(r'(?:FY)\s*(\d{4})', clean, re.IGNORECASE)
    if m:
        return date(int(m.group(1)), 12, 31)

    # "Q1 2024", "Q3'24", "Q2 '24"
    m = re.match(r'Q(\d)\s*[\'"]?(\d{2,4})', clean, re.IGNORECASE)
    if m:
        q = int(m.group(1))
        yr = m.group(2)
        year = int(yr) if len(yr) == 4 else 2000 + int(yr)
        month = {1: 3, 2: 6, 3: 9, 4: 12}.get(q, 12)
        return date(year, month, 28 if month == 2 else 30)

    # "H1 2024", "H2 2023"
    m = re.match(r'H(\d)\s*(\d{4})', clean, re.IGNORECASE)
    if m:
        h = int(m.group(1))
        year = int(m.group(2))
        return date(year, 6 if h == 1 else 12, 30)

    # "9M 2024"
    m = re.match(r'(\d+)M\s*(\d{4})', clean, re.IGNORECASE)
    if m:
        months = int(m.group(1))
        year = int(m.group(2))
        return date(year, min(months, 12), 28 if months == 2 else 30)

    # "Jan-24", "Sep-23", "Mar 2024"
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }
    m = re.match(r'(\w{3})\s*[-/]?\s*[\'"]?(\d{2,4})', clean, re.IGNORECASE)
    if m:
        mon_str = m.group(1).lower()
        yr = m.group(2)
        if mon_str in month_map:
            year = int(yr) if len(yr) == 4 else 2000 + int(yr)
            month = month_map[mon_str]
            return date(year, month, 28 if month == 2 else 30)

    # Plain year: "2024", "2023"
    m = re.match(r'^(\d{4})$', clean)
    if m:
        return date(int(m.group(1)), 12, 31)

    # Two-digit year: "'24", "24"
    m = re.match(r"^['\"]?(\d{2})$", clean)
    if m:
        year = 2000 + int(m.group(1))
        return date(year, 12, 31)

    return None


# ==============================================================================
# ORM MODEL
# ==============================================================================

class MetricFact(Base):
    """
    One row per DataPoint. Denormalized for fast SQL queries.
    """
    __tablename__ = "metric_facts"

    fact_id = Column(String, primary_key=True)
    doc_id = Column(String, nullable=False, index=True)
    doc_hash = Column(String, nullable=False, index=True)
    source_file = Column(String, nullable=False)
    page_number = Column(Float, nullable=False)  # int stored as float for SQLAlchemy compat
    item_id = Column(String, nullable=False)

    # Series-level fields (denormalized)
    series_label = Column(String, nullable=False, index=True)
    series_nature = Column(String, default="level")
    accounting_basis = Column(String, nullable=True)
    data_provenance = Column(String, nullable=True)
    periodicity = Column(String, nullable=True)
    source_region_id = Column(Float, nullable=True)

    # DataPoint-level fields
    label = Column(String, nullable=False)
    numeric_value = Column(Float, nullable=True)
    currency = Column(String, default="None")
    magnitude = Column(String, default="None")
    measure = Column(String, nullable=True)
    original_text = Column(String, default="")

    # Pre-computed fields (deterministic Python, never LLM)
    resolved_value = Column(Float, nullable=True)
    period_date = Column(Date, nullable=True)

    # Context
    category = Column(String, nullable=True)
    archetype = Column(String, nullable=True)
    confidence_score = Column(Float, default=0.0)
    indexed_at = Column(DateTime, default=datetime.utcnow)


# ==============================================================================
# DRIVER
# ==============================================================================

class AnalyticsDriver:
    """
    Manages the metric_facts table in Postgres.
    Provides SQL query execution for the Text-to-SQL pipeline.
    """

    def __init__(self):
        db_url = (
            f"postgresql://{SystemConfig.POSTGRES_USER}:{SystemConfig.POSTGRES_PASSWORD}"
            f"@{SystemConfig.POSTGRES_HOST}:{SystemConfig.POSTGRES_PORT}"
            f"/{SystemConfig.POSTGRES_DB}"
        )
        self.engine = create_engine(db_url, pool_pre_ping=True)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info("Analytics Driver initialized. Table: metric_facts")

    @traceable(name="Index MetricFacts", run_type="tool")
    def index_metrics(self, doc: UnifiedDocument) -> int:
        """
        Flattens all MetricSeries from VisualItems and ChartTableItems
        into metric_facts rows. Returns count of rows inserted.
        """
        doc_hash = doc.items[0].source.file_hash if doc.items else ""
        rows: List[MetricFact] = []

        for item in doc.items:
            series_list: List[MetricSeries] = []
            archetype_str: Optional[str] = None
            confidence: float = 0.0

            if isinstance(item, VisualItem):
                series_list = item.metrics
                archetype_str = (
                    item.archetype.value
                    if hasattr(item.archetype, 'value')
                    else str(item.archetype)
                )
            elif isinstance(item, ChartTableItem):
                series_list = item.visual_metrics
                archetype_str = (
                    item.archetype.value
                    if hasattr(item.archetype, 'value')
                    else str(item.archetype)
                )

            if not series_list:
                continue

            for series in series_list:
                for dp in series.data_points:
                    rows.append(MetricFact(
                        fact_id=str(uuid4()),
                        doc_id=doc.doc_id,
                        doc_hash=doc_hash,
                        source_file=doc.filename,
                        page_number=item.source.page_number,
                        item_id=item.id,
                        series_label=series.series_label,
                        series_nature=series.series_nature,
                        accounting_basis=series.accounting_basis,
                        data_provenance=series.data_provenance,
                        periodicity=(
                            dp.periodicity or series.periodicity
                        ),
                        source_region_id=series.source_region_id,
                        label=dp.label,
                        numeric_value=dp.numeric_value,
                        currency=dp.currency,
                        magnitude=dp.magnitude,
                        measure=dp.measure,
                        original_text=dp.original_text,
                        resolved_value=_resolve_value(dp.numeric_value, dp.magnitude),
                        period_date=_parse_period_date(dp.label),
                        category=None,
                        archetype=archetype_str,
                        confidence_score=confidence,
                    ))

        if not rows:
            logger.info(f"No metric facts to index for {doc.filename}")
            return 0

        session = self.SessionLocal()
        try:
            # Delete existing facts for this doc to allow re-ingestion
            session.execute(
                sql_text("DELETE FROM metric_facts WHERE doc_hash = :hash"),
                {"hash": doc_hash},
            )
            session.add_all(rows)
            session.commit()
            logger.info(f"Indexed {len(rows)} metric facts for {doc.filename}")
            return len(rows)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to index metric facts: {e}")
            raise
        finally:
            session.close()

    @traceable(name="Query MetricFacts", run_type="tool")
    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """
        Executes a read-only SQL query against metric_facts.
        Used by the Text-to-SQL pipeline.
        """
        session = self.SessionLocal()
        try:
            result = session.execute(sql_text(sql))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        finally:
            session.close()

    @traceable(name="Get Distinct Labels", run_type="tool")
    def get_distinct_labels(self) -> List[str]:
        """
        Returns all unique series_label values.
        Injected into Text-to-SQL prompts so the LLM knows exact label vocabulary.
        """
        session = self.SessionLocal()
        try:
            result = session.execute(
                sql_text("SELECT DISTINCT series_label FROM metric_facts ORDER BY series_label")
            )
            return [row[0] for row in result.fetchall()]
        finally:
            session.close()

    def delete_by_doc_hash(self, doc_hash: str) -> None:
        """Removes all facts for a given document hash."""
        session = self.SessionLocal()
        try:
            session.execute(
                sql_text("DELETE FROM metric_facts WHERE doc_hash = :hash"),
                {"hash": doc_hash},
            )
            session.commit()
            logger.info(f"Deleted metric facts for doc_hash: {doc_hash[:8]}...")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete metric facts: {e}")
            raise
        finally:
            session.close()

    def get_ddl(self) -> str:
        """
        Returns the CREATE TABLE DDL for metric_facts.
        Injected into Text-to-SQL prompts so the LLM knows the schema.
        """
        return """
CREATE TABLE metric_facts (
    fact_id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL,
    doc_hash VARCHAR NOT NULL,
    source_file VARCHAR NOT NULL,
    page_number FLOAT NOT NULL,
    item_id VARCHAR NOT NULL,
    series_label VARCHAR NOT NULL,
    series_nature VARCHAR DEFAULT 'level',
    accounting_basis VARCHAR,
    data_provenance VARCHAR,
    periodicity VARCHAR,
    source_region_id FLOAT,
    label VARCHAR NOT NULL,
    numeric_value DOUBLE PRECISION,
    currency VARCHAR DEFAULT 'None',
    magnitude VARCHAR DEFAULT 'None',
    measure VARCHAR,
    original_text VARCHAR DEFAULT '',
    resolved_value DOUBLE PRECISION,  -- PRE-COMPUTED: numeric_value * magnitude_multiplier
    period_date DATE,                 -- PRE-PARSED from label
    category VARCHAR,
    archetype VARCHAR,
    confidence_score REAL DEFAULT 0.0,
    indexed_at TIMESTAMP DEFAULT NOW()
);
-- IMPORTANT: resolved_value is the canonical query target.
-- It already incorporates magnitude (k=1e3, M=1e6, B=1e9, T=1e12).
-- Example: "revenue > $1B" → WHERE resolved_value > 1000000000
-- NEVER generate CASE statements for magnitude conversion in SQL.
""".strip()
