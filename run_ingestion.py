import asyncio
import logging
from src.config import SystemPaths
from src.ingestion.pipeline import IngestionPipeline

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ChironRunner")

async def main():
    """
    Manual trigger for the Chiron Ingestion Engine.
    Scans 'data/inputs/' and runs the pipeline for every file found.
    """
    pipeline = IngestionPipeline()
    
    # 1. Scan the Drop Zone
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
    for i, file_path in enumerate(input_files, 1):
        print(f"\n--- Processing {i}/{len(input_files)}: {file_path.name} ---")
        
        result = await pipeline.run(file_path)
        
        if result.status == "success":
            print(f"✅ SUCCESS: {file_path.name}")
            print(f"   - Chunks: {result.total_chunks}")
            print(f"   - Time:   {result.processing_time_seconds:.2f}s")
            print(f"   - Doc Hash: {result.doc_hash[:8]}...")
        else:
            print(f"❌ FAILED: {file_path.name}")
            print(f"   - Error: {result.error_msg}")

if __name__ == "__main__":
    asyncio.run(main())