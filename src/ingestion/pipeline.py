import logging
from pathlib import Path
from datetime import datetime
from langsmith import traceable

# Internal Components
from src.ingestion.router import SmartRouter, ProcessingRoute
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.native_parser import NativeParser
from src.storage.db_driver import ledger_db, DocumentLedger
from src.storage.vector_driver import VectorDriver
from src.schemas.documents import IngestionResult

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    The Orchestrator of the 'Dual-Helix' Engine (Centaur 2.0).
    Coordinates Routing -> Parsing -> Storage (Blobs/Ledger) -> Indexing.
    """

    def __init__(self):
        # 1. Initialize Parsers (The "Dual Helix")
        self.pdf_parser = PDFParser()      # Helix A: Visual/Complex
        self.native_parser = NativeParser() # Helix B: Native/Text
        
        # 2. Initialize Infrastructure
        self.router = SmartRouter()
        self.indexer = VectorDriver()
        
        logger.info("🚀 Centaur 2.0 Pipeline Initialized")

    @traceable(name="Run Ingestion Pipeline", run_type="chain")
    async def run(self, file_path: Path) -> IngestionResult:
        """
        Main entry point for processing a single file.
        """
        start_time = datetime.now()
        logger.info(f"📥 Pipeline triggered for: {file_path.name}")
        
        # 1. Routing (Helix Logic)
        # ------------------------------------------------------------------
        route, reason = self.router.route(file_path)
        
        if route == ProcessingRoute.UNSUPPORTED:
            logger.warning(f"⛔ Skipped {file_path.name}: {reason}")
            return self._build_result(file_path, start_time, status="skipped", error=reason)

        # 2. Execution (Parsing)
        # ------------------------------------------------------------------
        try:
            chunks = []
            
            # Route Handling: We explicitly check the Enum
            if route == ProcessingRoute.HELIX_A_VISUAL:
                chunks = await self.pdf_parser.parse(file_path)
            elif route == ProcessingRoute.HELIX_B_NATIVE:
                chunks = await self.native_parser.parse(file_path)
            else:
                raise ValueError(f"Route {route} not implemented")
            
            if not chunks:
                raise ValueError("Parser returned 0 chunks (Possible Empty File or OCR Failure)")
                
            doc_hash = chunks[0].doc_hash
            
            # 3. Indexing (The Search Truth)
            # ------------------------------------------------------------------
            logger.info(f"🧠 Indexing {len(chunks)} chunks to Qdrant...")
            self.indexer.index(chunks)
            
        except Exception as e:
            logger.exception(f"💥 Pipeline crashed on {file_path.name}")
            return self._build_result(file_path, start_time, status="failed", error=str(e))

        # 4. State Update (The Lineage Ledger)
        # ------------------------------------------------------------------
        # We record this ingestion event in Postgres to maintain the "Chain of Custody"
        if ledger_db:
            try:
                with ledger_db.session() as session:
                    # Check if doc exists to avoid duplicate key errors
                    existing = session.query(DocumentLedger).filter_by(doc_hash=doc_hash).first()
                    
                    if existing:
                        logger.info(f"📜 Document already exists in Ledger: {doc_hash[:8]}")
                        # Optional: Update 'last_processed_at' here
                    else:
                        # Calculate accurate VLM cost
                        # gpt-4o-mini is approx $0.15 per 1M input tokens
                        total_tokens = sum(c.token_count for c in chunks if c.token_count)
                        est_cost = (total_tokens / 1_000_000) * 0.15
                        
                        new_entry = DocumentLedger(
                            doc_hash=doc_hash,
                            filename=file_path.name,
                            status="completed",
                            processing_cost=est_cost
                        )
                        session.add(new_entry)
                        logger.info(f"✅ Registered in Lineage Ledger: {doc_hash[:8]}")
            except Exception as db_err:
                # DB errors should not crash the pipeline, just log them
                logger.error(f"⚠️ Ledger update failed: {db_err}")

        # 5. Result Summary
        # ------------------------------------------------------------------
        return self._build_result(
            file_path, start_time, status="success", chunks=chunks, doc_hash=doc_hash
        )

    def _build_result(self, file_path, start_time, status, chunks=None, doc_hash="N/A", error=None):
        """
        Constructs the standardized IngestionResult.
        Calculates the "Latte Benchmark" (Cost Estimate).
        """
        duration = (datetime.now() - start_time).total_seconds()
        total_chunks = len(chunks) if chunks else 0
        total_pages = chunks[-1].page_number if chunks else 0
        
        # Cost Estimation Logic (Centaur 2.0)
        # We assume 1 chunk ~ 1 page ~ 1 VLM call
        # gpt-4o-mini blended rate: ~$0.0004 per page (very cheap)
        # gpt-4o blended rate: ~$0.01 per page
        
        cost = 0.0
        if chunks:
            # Sum the actual token counts tracked by the Parser
            total_tokens = sum([c.token_count for c in chunks if c.token_count])
            # Price for gpt-4o-mini (Input: $0.15/1M, Output: $0.60/1M)
            # We assume 90% input (image), 10% output (JSON)
            cost = (total_tokens / 1_000_000) * 0.20 

        if status == "success":
             logger.info(f"🏁 Finished {file_path.name} in {duration:.2f}s. Cost: ${cost:.5f}")

        return IngestionResult(
            file_name=file_path.name,
            doc_hash=doc_hash,
            total_pages=total_pages,
            total_chunks=total_chunks,
            processing_time_seconds=duration,
            cost_estimate_usd=cost,
            status=status,
            error_msg=error
        )