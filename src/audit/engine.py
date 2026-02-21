"""
Audit Engine: Deterministic data quality validators.

Runs pure SQL checks against metric_facts after ingestion.
Zero LLM calls — all validators are deterministic code.

Validators:
1. StackedTotalCheck — stacked bar segment sums vs stated total
2. CrossPageConsistencyCheck — same metric across pages with different values
3. YoYReasonabilityCheck — extreme YoY growth (>300% or <-80%)
4. CurrencyMagnitudeCheck — mixed currencies/magnitudes within a series
5. PercentVerificationCheck — unreasonable percentage values
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Text, DateTime,
    text as sql_text,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from langsmith import traceable

from src.config import SystemConfig

logger = logging.getLogger(__name__)

Base = declarative_base()


# ==============================================================================
# ORM MODEL
# ==============================================================================

class AuditFinding(Base):
    """One row per audit finding. Idempotent — old findings deleted before re-audit."""
    __tablename__ = "audit_findings"

    id = Column(String, primary_key=True)
    doc_hash = Column(String, nullable=False, index=True)
    check_name = Column(String, nullable=False)
    severity = Column(String, nullable=False)       # error | warning | info
    title = Column(String, nullable=False)
    detail = Column(Text, default="")
    page_number = Column(Integer, nullable=True)
    series_label = Column(String, nullable=True)
    expected_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==============================================================================
# VALIDATOR INTERFACE
# ==============================================================================

class BaseAuditCheck(ABC):
    name: str

    @abstractmethod
    def run(self, session, doc_hash: str) -> List[AuditFinding]:
        ...


# ==============================================================================
# VALIDATORS
# ==============================================================================

class StackedTotalCheck(BaseAuditCheck):
    """For stacked bar charts with a 'total' series, verify sum of parts."""
    name = "stacked_total"

    def run(self, session, doc_hash: str) -> List[AuditFinding]:
        findings = []

        # Find pages with BAR archetype and a total series
        result = session.execute(sql_text("""
            SELECT DISTINCT page_number::int
            FROM metric_facts
            WHERE doc_hash = :doc_hash
              AND archetype ILIKE '%bar%'
              AND series_nature = 'total'
        """), {"doc_hash": doc_hash})

        pages_with_total = [row[0] for row in result.fetchall()]

        for page in pages_with_total:
            # Get total series values per label
            total_result = session.execute(sql_text("""
                SELECT label, resolved_value
                FROM metric_facts
                WHERE doc_hash = :doc_hash
                  AND page_number = :page
                  AND series_nature = 'total'
                  AND resolved_value IS NOT NULL
            """), {"doc_hash": doc_hash, "page": page})
            totals = {row[0]: row[1] for row in total_result.fetchall()}

            # Sum non-total series per label
            sum_result = session.execute(sql_text("""
                SELECT label, SUM(resolved_value) as computed_total
                FROM metric_facts
                WHERE doc_hash = :doc_hash
                  AND page_number = :page
                  AND series_nature != 'total'
                  AND resolved_value IS NOT NULL
                GROUP BY label
            """), {"doc_hash": doc_hash, "page": page})

            for row in sum_result.fetchall():
                label, computed = row[0], row[1]
                if label in totals and totals[label] and computed:
                    stated = totals[label]
                    if abs(stated) > 0 and abs(computed - stated) / abs(stated) > 0.02:
                        findings.append(AuditFinding(
                            id=str(uuid4()),
                            doc_hash=doc_hash,
                            check_name=self.name,
                            severity="error",
                            title=f"Stacked total mismatch on Page {page}",
                            detail=(
                                f"For '{label}': segments sum to {computed:,.2f} "
                                f"but total shows {stated:,.2f} "
                                f"(diff: {abs(computed - stated):,.2f})"
                            ),
                            page_number=page,
                            series_label=label,
                            expected_value=computed,
                            actual_value=stated,
                        ))

        return findings


class CrossPageConsistencyCheck(BaseAuditCheck):
    """Flag same metric + period on multiple pages with different values."""
    name = "cross_page_consistency"

    def run(self, session, doc_hash: str) -> List[AuditFinding]:
        findings = []

        result = session.execute(sql_text("""
            SELECT series_label, label,
                   COUNT(DISTINCT page_number::int) as page_count,
                   COUNT(DISTINCT ROUND(resolved_value::numeric, 2)) as value_count,
                   MIN(resolved_value) as min_val,
                   MAX(resolved_value) as max_val,
                   STRING_AGG(DISTINCT page_number::int::text, ', '
                              ORDER BY page_number::int::text) as pages
            FROM metric_facts
            WHERE doc_hash = :doc_hash
              AND resolved_value IS NOT NULL
            GROUP BY series_label, label
            HAVING COUNT(DISTINCT page_number::int) > 1
               AND COUNT(DISTINCT ROUND(resolved_value::numeric, 2)) > 1
        """), {"doc_hash": doc_hash})

        for row in result.fetchall():
            series, label, page_count, value_count, min_val, max_val, pages = row
            findings.append(AuditFinding(
                id=str(uuid4()),
                doc_hash=doc_hash,
                check_name=self.name,
                severity="warning",
                title=f"'{series}' inconsistent across pages",
                detail=(
                    f"Period '{label}' has {value_count} different values "
                    f"across pages [{pages}]: "
                    f"range [{min_val:,.2f}, {max_val:,.2f}]"
                ),
                page_number=None,
                series_label=series,
                expected_value=min_val,
                actual_value=max_val,
            ))

        return findings


class YoYReasonabilityCheck(BaseAuditCheck):
    """Flag extreme YoY growth (>300% or <-80%) between consecutive periods."""
    name = "yoy_reasonability"

    def run(self, session, doc_hash: str) -> List[AuditFinding]:
        findings = []

        result = session.execute(sql_text("""
            WITH ordered AS (
                SELECT series_label, label, period_date, resolved_value,
                       page_number::int as page,
                       LAG(resolved_value) OVER (
                           PARTITION BY series_label ORDER BY period_date
                       ) as prev_value,
                       LAG(label) OVER (
                           PARTITION BY series_label ORDER BY period_date
                       ) as prev_label
                FROM metric_facts
                WHERE doc_hash = :doc_hash
                  AND period_date IS NOT NULL
                  AND resolved_value IS NOT NULL
                  AND series_nature = 'level'
            )
            SELECT series_label, prev_label, label, prev_value,
                   resolved_value, page
            FROM ordered
            WHERE prev_value IS NOT NULL
              AND ABS(prev_value) > 0
              AND (
                  (resolved_value - prev_value) / ABS(prev_value) > 3.0
                  OR (resolved_value - prev_value) / ABS(prev_value) < -0.8
              )
        """), {"doc_hash": doc_hash})

        for row in result.fetchall():
            series, prev_label, label, prev_val, curr_val, page = row
            growth = (curr_val - prev_val) / abs(prev_val) * 100
            findings.append(AuditFinding(
                id=str(uuid4()),
                doc_hash=doc_hash,
                check_name=self.name,
                severity="warning",
                title=f"Extreme change in '{series}'",
                detail=(
                    f"From '{prev_label}' to '{label}': "
                    f"{growth:+,.1f}% change "
                    f"({prev_val:,.2f} -> {curr_val:,.2f})"
                ),
                page_number=page,
                series_label=series,
                expected_value=prev_val,
                actual_value=curr_val,
            ))

        return findings


class CurrencyMagnitudeCheck(BaseAuditCheck):
    """Flag series mixing currencies or magnitudes — usually a VLM extraction error."""
    name = "currency_magnitude"

    def run(self, session, doc_hash: str) -> List[AuditFinding]:
        findings = []

        # Check mixed currencies
        result = session.execute(sql_text("""
            SELECT series_label,
                   STRING_AGG(DISTINCT currency, ', ') as currencies,
                   STRING_AGG(DISTINCT magnitude, ', ') as magnitudes,
                   COUNT(DISTINCT currency) as curr_ct,
                   COUNT(DISTINCT magnitude) as mag_ct,
                   MIN(page_number)::int as page
            FROM metric_facts
            WHERE doc_hash = :doc_hash
              AND currency != 'None'
            GROUP BY series_label
            HAVING COUNT(DISTINCT currency) > 1
                OR (COUNT(DISTINCT magnitude) > 1
                    AND COUNT(DISTINCT magnitude) FILTER (WHERE magnitude != 'None') > 1)
        """), {"doc_hash": doc_hash})

        for row in result.fetchall():
            series, currencies, magnitudes, curr_ct, mag_ct, page = row
            issues = []
            if curr_ct > 1:
                issues.append(f"mixed currencies: [{currencies}]")
            if mag_ct > 1:
                issues.append(f"mixed magnitudes: [{magnitudes}]")

            if issues:
                findings.append(AuditFinding(
                    id=str(uuid4()),
                    doc_hash=doc_hash,
                    check_name=self.name,
                    severity="error",
                    title=f"Unit inconsistency in '{series}'",
                    detail="; ".join(issues),
                    page_number=page,
                    series_label=series,
                ))

        return findings


class PercentVerificationCheck(BaseAuditCheck):
    """Flag percentage series with unreasonable values for level metrics."""
    name = "percent_verification"

    def run(self, session, doc_hash: str) -> List[AuditFinding]:
        findings = []

        result = session.execute(sql_text("""
            SELECT series_label, label, resolved_value, page_number::int
            FROM metric_facts
            WHERE doc_hash = :doc_hash
              AND measure = '%%'
              AND series_nature = 'level'
              AND resolved_value IS NOT NULL
              AND ABS(resolved_value) > 1000
        """), {"doc_hash": doc_hash})

        for row in result.fetchall():
            series, label, value, page = row
            findings.append(AuditFinding(
                id=str(uuid4()),
                doc_hash=doc_hash,
                check_name=self.name,
                severity="info",
                title=f"Unusual percentage in '{series}'",
                detail=(
                    f"Period '{label}': {value:,.1f}% seems abnormally high "
                    f"for a level metric"
                ),
                page_number=page,
                series_label=series,
                actual_value=value,
            ))

        return findings


# ==============================================================================
# ENGINE ORCHESTRATOR
# ==============================================================================

class AuditEngine:
    """Orchestrates all audit checks against metric_facts."""

    checks: List[BaseAuditCheck] = [
        StackedTotalCheck(),
        CrossPageConsistencyCheck(),
        YoYReasonabilityCheck(),
        CurrencyMagnitudeCheck(),
        PercentVerificationCheck(),
    ]

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
        logger.info("Audit Engine initialized. Table: audit_findings")

    @traceable(name="Run Audit Checks", run_type="tool")
    def run_all(self, doc_hash: str) -> List[Dict[str, Any]]:
        """Run all checks, store findings, return results as dicts."""
        session = self.SessionLocal()
        try:
            # Delete old findings for idempotent re-audit
            session.execute(
                sql_text("DELETE FROM audit_findings WHERE doc_hash = :hash"),
                {"hash": doc_hash},
            )

            all_findings: List[AuditFinding] = []
            for check in self.checks:
                try:
                    results = check.run(session, doc_hash)
                    all_findings.extend(results)
                    if results:
                        logger.info(
                            f"Audit [{check.name}]: {len(results)} findings "
                            f"for {doc_hash[:8]}"
                        )
                except Exception as e:
                    logger.error(f"Audit [{check.name}] failed: {e}", exc_info=True)

            # Bulk insert
            if all_findings:
                session.add_all(all_findings)
            session.commit()

            logger.info(
                f"Audit complete: {len(all_findings)} findings for {doc_hash[:8]}"
            )
            return [_finding_to_dict(f) for f in all_findings]
        except Exception as e:
            session.rollback()
            logger.error(f"Audit engine failed: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def get_findings(self, doc_hash: str) -> List[Dict[str, Any]]:
        """Retrieve cached findings for a document."""
        session = self.SessionLocal()
        try:
            result = session.execute(
                sql_text("""
                    SELECT id, check_name, severity, title, detail,
                           page_number, series_label, expected_value, actual_value
                    FROM audit_findings
                    WHERE doc_hash = :hash
                    ORDER BY
                        CASE severity
                            WHEN 'error' THEN 0
                            WHEN 'warning' THEN 1
                            ELSE 2
                        END,
                        page_number NULLS LAST
                """),
                {"hash": doc_hash},
            )
            return [
                {
                    "id": row[0],
                    "check_name": row[1],
                    "severity": row[2],
                    "title": row[3],
                    "detail": row[4],
                    "page_number": row[5],
                    "series_label": row[6],
                    "expected_value": row[7],
                    "actual_value": row[8],
                }
                for row in result.fetchall()
            ]
        finally:
            session.close()

    def get_summary(self, doc_hash: str) -> Dict[str, int]:
        """Return severity counts for a document."""
        session = self.SessionLocal()
        try:
            result = session.execute(
                sql_text("""
                    SELECT severity, COUNT(*) as cnt
                    FROM audit_findings
                    WHERE doc_hash = :hash
                    GROUP BY severity
                """),
                {"hash": doc_hash},
            )
            counts = {"error": 0, "warning": 0, "info": 0}
            for row in result.fetchall():
                counts[row[0]] = row[1]
            return counts
        finally:
            session.close()


def _finding_to_dict(f: AuditFinding) -> Dict[str, Any]:
    return {
        "id": f.id,
        "check_name": f.check_name,
        "severity": f.severity,
        "title": f.title,
        "detail": f.detail,
        "page_number": f.page_number,
        "series_label": f.series_label,
        "expected_value": f.expected_value,
        "actual_value": f.actual_value,
    }
