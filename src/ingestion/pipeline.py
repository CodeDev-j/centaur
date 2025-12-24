import logging
from pathlib import Path
from datetime import datetime
from langsmith import traceable

# Internal Components
from src.ingestion.router import SmartRouter, ProcessingRoute
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.native_parser import NativeParser
from src.schemas.documents import IngestionResult
from src.storage.db_driver import ledger_db, DocumentLedger
from src.storage.vector_driver import VectorDriver  # <--- NEW: The Indexer

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    The Orchestrator of the 'Dual-Helix' Engine.
    Coordinates Routing -> Parsing -> Storage (Blobs/Ledger) -> Indexing (Qdrant).
    """

    def __init__(self):
        self.router = SmartRouter()
        self.pdf_parser = PDFParser()
        self.native_parser = NativeParser()
        self.indexer = VectorDriver()  # <--- NEW: Initialize Vector Driver
        
    @traceable(name="Run Ingestion Pipeline", run_type="chain")
    async def run(self, file_path: Path) -> IngestionResult:
        """
        Main entry point for processing a single file.
        """
        start_time = datetime.now()
        logger.info(f"🚀 Pipeline triggered for: {file_path.name}")
        
        # 1. Routing
        # ------------------------------------------------------------------
        route, reason = self.router.route(file_path)
        
        if route == ProcessingRoute.UNSUPPORTED:
            logger.warning(f"⛔ Skipped {file_path.name}: {reason}")
            return IngestionResult(
                file_name=file_path.name,
                doc_hash="N/A",
                total_pages=0,
                total_chunks=0,
                processing_time_seconds=0,
                cost_estimate_usd=0,
                status="skipped",
                error_msg=reason
            )

        # 2. Execution (The Dual Helix)
        # ------------------------------------------------------------------
        try:
            if route == ProcessingRoute.HELIX_A_VISUAL:
                chunks = await self.pdf_parser.parse(file_path)
            elif route == ProcessingRoute.HELIX_B_NATIVE:
                chunks = await self.native_parser.parse(file_path)
            else:
                raise ValueError(f"Route {route} not implemented")
            
            if not chunks:
                raise ValueError("Parser returned 0 chunks")
                
            doc_hash = chunks[0].doc_hash
            
            # 3. Indexing (The Search Truth)
            # ------------------------------------------------------------------
            # We now push the clean chunks to Qdrant so the Brain can find them.
            logger.info("🧠 Generating Embeddings & Indexing...")
            self.indexer.index(chunks)
            
        except Exception as e:
            logger.error(f"💥 Pipeline crashed on {file_path.name}: {e}")
            return IngestionResult(
                file_name=file_path.name,
                doc_hash="error",
                total_pages=0,
                total_chunks=0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                cost_estimate_usd=0,
                status="failed",
                error_msg=str(e)
            )

        # 4. State Update (The Lineage Ledger)
        # ------------------------------------------------------------------
        # We record this ingestion event in Postgres to prevent "Hallucinating Old Data"
        if ledger_db:
            try:
                with ledger_db.session() as session:
                    # Check if this doc version already exists
                    existing = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                    
                    if existing:
                        logger.info(f"📜 Document already exists in Ledger: {doc_hash[:8]}")
                        # In a real system, we might update the 'last_accessed' timestamp here
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
        logger.info(f"🏁 Finished {file_path.name} in {duration:.2f}s. Generated {len(chunks)} chunks.")
        
        return IngestionResult(
            file_name=file_path.name,
            doc_hash=doc_hash,
            total_pages=chunks[-1].page_number if chunks else 0,
            total_chunks=len(chunks),
            processing_time_seconds=duration,
            cost_estimate_usd=0.0,
            status="success"
        )