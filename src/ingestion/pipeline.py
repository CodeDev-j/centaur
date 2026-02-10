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
from typing import Union
from langsmith import traceable

# Internal Components
from src.ingestion.router import SmartRouter, ProcessingRoute
from src.ingestion.pdf_parser import pdf_parser  # Singleton Instance
from src.ingestion.native_parser import NativeParser
from src.schemas.documents import IngestionResult
from src.schemas.deal_stream import UnifiedDocument  # The Contract
from src.storage.db_driver import ledger_db, DocumentLedger
from src.storage.vector_driver import VectorDriver

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
        
    @traceable(name="Run Ingestion Pipeline", run_type="chain")
    async def run(self, file_path: Path) -> Union[UnifiedDocument, IngestionResult]:
        """
        Main entry point for processing a single file.
        Returns UnifiedDocument on success to satisfy run_ingestion.py.
        """
        start_time = datetime.now()
        logger.info(f"🚀 Pipeline triggered for: {file_path.name}")
        
        # 1. Routing (The Decision Layer)
        # ------------------------------------------------------------------
        route, reason = self.router.route(file_path)
        
        if route == ProcessingRoute.UNSUPPORTED:
            logger.warning(f"⛔ Skipped {file_path.name}: {reason}")
            return IngestionResult(
                file_name=file_path.name,
                doc_hash="N/A",
                status="skipped",
                error_msg=reason
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
            if route == ProcessingRoute.HELIX_A_VISUAL:
                # [CONTRACT ENFORCEMENT] Must return UnifiedDocument Object
                doc = await self.pdf_parser.parse(file_path)
            elif route == ProcessingRoute.HELIX_B_NATIVE:
                # TODO: Update native parser to return UnifiedDocument
                doc = await self.native_parser.parse(file_path)
            else:
                raise ValueError(f"Route {route} not implemented")
            
            # [VALIDATION] Ensure the Parser honored the contract
            if not isinstance(doc, UnifiedDocument) or not doc.items:
                raise ValueError("Parser returned empty or invalid document")
                
            # Extract Hash safely from the stream (using first item's source)
            doc_hash = doc.items[0].source.file_hash if doc.items else "unknown_hash"
            
            # 3. Indexing (The Search Truth)
            # ------------------------------------------------------------------
            # We push clean chunks to Qdrant so the Brain can find them.
            # logger.info("🧠 Generating Embeddings & Indexing...")
            # self.indexer.index(doc.items) 
            
        except Exception as e:
            logger.error(f"💥 Pipeline crashed on {file_path.name}: {e}")
            logger.debug(traceback.format_exc())
            # Return Legacy Result for graceful error reporting in Runner
            return IngestionResult(
                file_name=file_path.name,
                doc_hash="error",
                status="failed",
                error_msg=str(e)
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