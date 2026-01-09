import argparse
import json
import logging
import textwrap
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
# We connect directly to Qdrant to ensure we see "Raw Truth" for debugging
DB_URL = "http://localhost:6333"
COLLECTION_NAME = "chiron_knowledge_base"

# Defaults
DEFAULT_FILENAME = "Alphabet_2025.Q1_Earnings.Slides - Copy-05.pdf"
DEFAULT_PAGE = 1
# ==============================================================================

# Configure logging to only show errors, keeping stdout clean for our tool
logging.basicConfig(level=logging.ERROR)

class DBInspector:
    def __init__(self):
        self.client = QdrantClient(url=DB_URL)
        self.collection = COLLECTION_NAME

    def list_files(self):
        """Scans DB to find all unique source files."""
        print(f"📂 Scanning {self.collection} for active files...")
        try:
            # Fetch a batch of points to find unique sources
            response = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]

            unique_files = set()
            for point in response:
                # Check ROOT level for 'source' (Flat Schema)
                if "source" in point.payload:
                    unique_files.add(point.payload["source"])
            
            print(f"\n✅ Found {len(unique_files)} active files:")
            for src in sorted(unique_files):
                print(f"   - {src}")
                
        except Exception as e:
            print(f"❌ Error scanning DB: {e}")

    def inspect_visuals(self):
        """Filters for VLM artifacts (charts/figures)."""
        print(f"🔍 Scanning {self.collection} for VLM artifacts...\n")
        
        results = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type", # Flat schema
                        match=models.MatchValue(value="visual")
                    )
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0] 

        if not results:
            print("❌ No visual chunks found! (Check if VLM ran)")
            return

        self._print_results(results, mode="visual")

    def inspect_page(self, filename: str, page_number: int):
        """Retrieves all chunks for a specific page."""
        print(f"\n🔍 Querying: '{filename}' (Page {page_number})...")

        results = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source", 
                        match=models.MatchValue(value=filename)
                    ),
                    models.FieldCondition(
                        key="page", 
                        match=models.MatchValue(value=page_number)
                    )
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False
        )[0]

        if not results:
            print("❌ No documents found.")
            return

        print(f"✅ Found {len(results)} chunk(s) on this page.\n")
        self._print_results(results, mode="page")

    def _print_results(self, results, mode="page"):
        """Helper to pretty-print chunks uniformly."""
        for i, point in enumerate(results):
            p = point.payload
            chunk_type = p.get("type", "unknown").upper()
            
            print(f"--- 📄 CHUNK {i + 1} [{chunk_type}] ---")
            
            if mode == "visual":
                print(f"📂 Source: {p.get('source', 'Unknown')}")
                print(f"🖼️ Image: {p.get('image_path', 'No Path')}")
            
            # Metadata Dump (Exclude massive text fields for clarity)
            display_meta = {k: v for k, v in p.items() if k not in ['text', 'ocr_evidence', 'html']}
            print("METADATA:")
            print(json.dumps(display_meta, indent=4))

            # Content Preview
            raw_text = p.get("clean_text") or p.get("text") or ""
            print("\nCONTENT:")
            print("=" * 60)
            print(textwrap.fill(raw_text[:2000], width=80)) # Limit chars for sanity
            if len(raw_text) > 2000:
                print("... [truncated]")
            print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centaur DB Inspector")
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all ingested files")
    group.add_argument("--visuals", action="store_true", help="Inspect extracted charts/images")
    group.add_argument("--page", type=int, help="Inspect a specific page number")
    
    # Optional file override
    parser.add_argument("--file", type=str, default=DEFAULT_FILENAME, help="Target filename (for page inspection)")

    args = parser.parse_args()
    inspector = DBInspector()

    if args.list:
        inspector.list_files()
    elif args.visuals:
        inspector.inspect_visuals()
    elif args.page is not None:
        inspector.inspect_page(args.file, args.page)