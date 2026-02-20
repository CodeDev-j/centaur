"""
Ingestion Pipeline: The Middleware Authority.

GOAL: This module acts as the "Central Nervous System" for the ingestion process.
It enforces the architectural boundaries between the "Runner" (Execution Layer) 
and the "Parsers" (Logic Layer).

COGNITIVE CONTRACT:
1.  **Dual-Helix Routing:** It intelligently forks execution based on file type 
    (Visual/PDF vs. Native/Excel).
    *Note: The architecture is pre-wired to support a future 'Legal Fork' 
    (Credit Agreements/LMA) distinct from the current 'Financial Fork' (CIMs).*
2.  **Strict Typing:** It serves as the gateway that guarantees `run_ingestion.py` 
    receives a compliant `UnifiedDocument` object, preventing loose dictionaries 
    from leaking upstream.
3.  **State Consistency:** It manages the side-effects of ingestion (Ledger Recording, 
    Vector Indexing) so that the Parsers remain pure and stateless.
"""

import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Union
from langsmith import traceable

# Progress callback signature: (phase: str, detail: str) -> None
ProgressCallback = Callable[[str, str], None]

# Internal Components
from src.ingestion.router import SmartRouter, ProcessingRoute
from src.ingestion.pdf_parser import pdf_parser  # Singleton Instance
from src.ingestion.native_parser import NativeParser
from src.schemas.documents import IngestionResult
from src.schemas.deal_stream import UnifiedDocument  # The Contract
from src.storage.db_driver import ledger_db, DocumentLedger
from src.storage.vector_driver import VectorDriver
from src.storage.analytics_driver import AnalyticsDriver
from src.ingestion.chunker import flatten_document

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    The Orchestrator of the 'Dual-Helix' Engine.
    Coordinates Routing -> Parsing -> Storage (Blobs/Ledger) -> Indexing (Qdrant).
    """

    def __init__(self):
        self.router = SmartRouter()
        self.pdf_parser = pdf_parser
        self.native_parser = NativeParser()
        self.indexer = VectorDriver()
        self.analytics = AnalyticsDriver()
        
    @traceable(name="Run Ingestion Pipeline", run_type="chain")
    async def run(
        self,
        file_path: Path,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Union[UnifiedDocument, IngestionResult]:
        """
        Main entry point for processing a single file.
        Returns UnifiedDocument on success to satisfy run_ingestion.py.

        on_progress: Optional callback (phase, detail) for SSE progress reporting.
        """
        start_time = datetime.now()
        logger.info(f"🚀 Pipeline triggered for: {file_path.name}")
        _emit = on_progress or (lambda phase, detail: None)
        
        # 1. Routing (The Decision Layer)
        # ------------------------------------------------------------------
        route, reason = self.router.route(file_path)
        
        if route == ProcessingRoute.UNSUPPORTED:
            logger.warning(f"⛔ Skipped {file_path.name}: {reason}")
            return IngestionResult(
                file_name=file_path.name,
                doc_hash="N/A",
                status="skipped",
                error_msg=reason,
                total_pages=0,
                total_chunks=0,
                processing_time_seconds=0.0,
                cost_estimate_usd=0.0
            )

        # ==============================================================================
        # 🚦 FUTURE STATE: ROUTING LOGIC (The "Legal Parser" Ambition)
        # ==============================================================================
        # CURRENT:
        #   We assume all inputs are Financial Artifacts (CIMs, Lender Pres, Models).
        #   We route everything to 'PDFParser' which specializes in:
        #   - Table Extraction (Docling)
        #   - Chart/Visual Analysis (VLM)
        #
        # FUTURE (Post-MVP):
        #   1. Classification Step: Detect if doc is "Credit Agreement" / "LMA" style.
        #   2. If Legal: Route to new 'LegalParser'.
        #      - FOCUS: Section 1.01 Defined Terms, Negative Covenants, Cross-Refs.
        #      - STORAGE: Will revive 'blobs/definitions' for massive definition blocks.
        #      - STRATEGY: Regex/NLP heavy, less visual.
        # ==============================================================================

        # 2. Execution (The Dual Helix)
        # ------------------------------------------------------------------
        try:
            _emit("parsing", f"Parsing {file_path.name}...")
            if route == ProcessingRoute.HELIX_A_VISUAL:
                doc = await self.pdf_parser.parse(file_path)
            elif route == ProcessingRoute.HELIX_B_NATIVE:
                doc = await self.native_parser.parse(file_path)
            else:
                raise ValueError(f"Route {route} not implemented")

            if not isinstance(doc, UnifiedDocument) or not doc.items:
                raise ValueError("Parser returned empty or invalid document")

            n_items = len(doc.items)
            n_quarantined = len(doc.quarantined_items) if doc.quarantined_items else 0
            _emit("parsed", f"{n_items} items extracted, {n_quarantined} quarantined")

            doc_hash = doc.items[0].source.file_hash if doc.items else "unknown_hash"

            # 3. Indexing
            _emit("indexing", "Chunking and embedding...")
            logger.info("Cleaning old vectors & indexing to Qdrant...")
            await self.indexer.delete_by_doc_hash(doc_hash)

            chunks = flatten_document(doc)
            if chunks:
                _emit("indexing", f"Embedding {len(chunks)} chunks...")
                await self.indexer.index(chunks)

            _emit("indexing", "Writing metric facts to Postgres...")
            logger.info("Indexing metric facts to Postgres...")
            self.analytics.index_metrics(doc)
            
        except Exception as e:
            logger.error(f"💥 Pipeline crashed on {file_path.name}: {e}")
            logger.debug(traceback.format_exc())
            # Return Legacy Result for graceful error reporting in Runner
            # Includes default zeros to prevent Pydantic validation errors
            return IngestionResult(
                file_name=file_path.name,
                doc_hash="error",
                status="failed",
                error_msg=str(e),
                total_pages=0,
                total_chunks=0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                cost_estimate_usd=0.0
            )

        # 4. State Update (The Lineage Ledger)
        # ------------------------------------------------------------------
        if ledger_db:
            try:
                with ledger_db.session() as session:
                    existing = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                    
                    if existing:
                        logger.info(f"📜 Document already exists in Ledger: {doc_hash[:8]}")
                    else:
                        new_entry = DocumentLedger(
                            doc_hash=doc_hash,
                            filename=file_path.name,
                            status="completed",
                            processing_cost=0.0 # TODO: Calculate based on token count
                        )
                        session.add(new_entry)
                        logger.info(f"✅ Registered in Lineage Ledger: {doc_hash[:8]}")
            except Exception as db_err:
                logger.error(f"⚠️ Ledger update failed (non-critical): {db_err}")

        # 5. Result Summary
        # ------------------------------------------------------------------
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"🏁 Finished {file_path.name} in {duration:.2f}s. Generated {len(doc.items)} items.")
        
        # [CONTRACT FULFILLMENT] Return the UnifiedDocument object directly
        return doc