"""
Chiron Ingestion Runner: The Manual Trigger
===========================================

1. THE MISSION
--------------
To serve as the primary entry point for manual ingestion jobs. It scans the
`data/inputs/` directory and feeds every file found into the `IngestionPipeline`.

2. THE MECHANISM
----------------
- **Discovery:** Iterates through the local input folder (ignoring hidden files).
- **Execution:** Runs the async pipeline for each file sequentially.
- **Reporting:** Provides "Dual Output" visibility:
    1. Console (Print): Immediate visual feedback for the operator.
    2. Logs (File/Stream): Permanent audit trail for debugging.

3. THE CONTRACT
---------------
- **Resilience:** A crash in one file MUST NOT stop the batch. The runner wraps
  each execution in a broad `try/except` block to ensure the queue finishes.
- **Transparency:** Reports both Success metrics (Items generated) and Failure
  metrics (Quarantined items) so operators trust the result.
"""

import asyncio
import logging
import time
import warnings

# --- Internal Core Components ---
from src.config import SystemPaths
from src.ingestion.pipeline import IngestionPipeline
from src.schemas.deal_stream import UnifiedDocument

# ------------------------------------------------------------------------------
# 🔇 WARNING SUPPRESSION
# ------------------------------------------------------------------------------
# Suppress Pydantic warnings about "model_" fields in third-party libraries
# (e.g., Docling, LangChain) that conflict with Pydantic v2 protected namespaces.
warnings.filterwarnings(
    "ignore",
    message=".*Field \"model_.*\" has conflict with protected namespace.*"
)

# ------------------------------------------------------------------------------
# ⚙️ LOGGING CONFIGURATION
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ChironRunner")


async def main():
    """
    Main Execution Loop.
    Scans 'data/inputs/' and runs the pipeline for every file found.
    """
    pipeline = IngestionPipeline()
    
    # 1. Scan the Drop Zone
    # --------------------------------------------------------------------------
    if not SystemPaths.INPUTS.exists():
        logger.error(f"❌ Input directory missing: {SystemPaths.INPUTS}")
        return

    input_files = [
        f for f in SystemPaths.INPUTS.iterdir() 
        if f.is_file() and not f.name.startswith(".") # Ignore hidden files
    ]
    
    if not input_files:
        logger.warning(f"⚠️ No files found in {SystemPaths.INPUTS}")
        print("\n👉 ACTION: Please drop a PDF or Excel file into 'data/inputs/' and run this again.\n")
        return

    logger.info(f"📂 Found {len(input_files)} files to process.")

    # 2. Process Queue
    # --------------------------------------------------------------------------
    for i, file_path in enumerate(input_files, 1):
        # [DUAL OUTPUT] Print for console visibility, Log for record keeping
        msg = f"--- Processing {i}/{len(input_files)}: {file_path.name} ---"
        logger.info(msg)
        print(f"\n{msg}")
        
        start_time = time.time()
        
        try:
            # Execute Pipeline
            result = await pipeline.run(file_path)
            duration = time.time() - start_time

            # [BRANCH A] Unified Deal Theory Output (The Gold Standard)
            if isinstance(result, UnifiedDocument):
                # Log structured summary
                logger.info(
                    f"✅ SUCCESS: {file_path.name} | "
                    f"DocID: {result.doc_id[:8]} | "
                    f"Items: {len(result.items)} | "
                    f"Errors: {len(result.quarantined_items)} | "
                    f"Time: {duration:.2f}s"
                )
                
                # Print visual summary for Operator
                print(f"✅ SUCCESS: {file_path.name}")
                print(f"   - Items:  {len(result.items)}")           # Valid data
                print(f"   - Errors: {len(result.quarantined_items)}") # Soft failures
                print(f"   - Time:   {duration:.2f}s")
                print(f"   - Doc ID: {result.doc_id[:8]}...")
                print(f"   - Type:   {result.signals.detected_artifact_type}")

                # Inspect Quarantine (Debug Visibility)
                if result.quarantined_items:
                    print("\n   🔍 QUARANTINE REPORT:")
                    for idx, q_item in enumerate(result.quarantined_items, 1):
                        q_type = q_item.get('type', 'Unknown')
                        q_err = q_item.get('error', 'No message')
                        print(f"     {idx}. Type:  {q_type}")
                        print(f"        Error: {q_err}")
            
            # [BRANCH B] Legacy/Fallback Output (Safety Net)
            elif hasattr(result, "status") and result.status == "success":
                logger.info(f"✅ SUCCESS (Legacy): {file_path.name}")
                print(f"✅ SUCCESS: {file_path.name}")
                print(f"   - Time:   {duration:.2f}s")
            
            # [BRANCH C] Explicit Failure
            else:
                # Handle explicit failures or unknown types
                error_msg = getattr(result, "error_msg", f"Unknown type: {type(result)}")
                logger.error(f"❌ FAILED: {file_path.name} - {error_msg}")
                print(f"❌ FAILED: {file_path.name}")
                print(f"   - Error: {error_msg}")

        except Exception as e:
            # [SAFETY NET] Catch-all for pipeline crashes.
            # Ensures one bad file doesn't kill the entire batch.
            logger.error(f"❌ CRASHED: {file_path.name}", exc_info=True)
            print(f"❌ CRASHED: {file_path.name}")
            print(f"   - Exception: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())