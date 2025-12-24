import logging
import os
from typing import List, Iterable
from uuid import uuid4

# Third-party
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from langsmith import traceable

# Internal
from src.schemas.documents import IngestedChunk

logger = logging.getLogger(__name__)

class VectorDriver:
    """
    Helix C: The Indexer.
    Responsible for the 'Search Truth'.
    1. Embeds text locally (FastEmbed) -> No API costs.
    2. Upserts vectors to Qdrant (Docker).
    """

    # We use a specific collection name for the credit knowledge base
    COLLECTION_NAME = "chiron_knowledge_base"
    
    # FastEmbed Model (Local, CPU-optimized, SOTA performance)
    # BAAI/bge-small-en-v1.5 is the current gold standard for local RAG
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        # connect to Qdrant running in Docker
        self.client = QdrantClient(host="localhost", port=6333)
        
        # Initialize the embedding model (downloads automatically on first run)
        logger.info(f"🧠 Loading Embedding Model: {self.EMBEDDING_MODEL_NAME}...")
        self.embedding_model = TextEmbedding(model_name=self.EMBEDDING_MODEL_NAME)
        
        # Ensure the collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Idempotent check to create the collection if it doesn't exist.
        """
        if not self.client.collection_exists(self.COLLECTION_NAME):
            logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384,  # bge-small-en-v1.5 dimension
                    distance=models.Distance.COSINE
                )
            )

    @traceable(name="Index Chunks", run_type="tool")
    def index(self, chunks: List[IngestedChunk]) -> bool:
        """
        Main entry point. Takes parsed chunks, embeds them, and saves to Qdrant.
        """
        if not chunks:
            return False

        try:
            texts = [chunk.clean_text for chunk in chunks]
            
            # 1. Generate Embeddings (Local CPU)
            # FastEmbed handles batching automatically
            embeddings = list(self.embedding_model.embed(texts))
            
            # 2. Prepare Payload (Metadata)
            points = []
            for i, chunk in enumerate(chunks):
                # We store minimal metadata in Qdrant.
                # Full content is in Blobs, but we keep text here for simpler retrieval.
                payload = {
                    "source": chunk.metadata.get("source"),
                    "page": chunk.page_number,
                    "type": chunk.metadata.get("type", "text"),
                    "text": chunk.clean_text, # Storing text allows retrieval without blob lookup
                    "doc_hash": chunk.doc_hash,
                    "chunk_id": chunk.chunk_id
                }
                
                points.append(models.PointStruct(
                    id=chunk.chunk_id, # We use the UUID generated in the parser
                    vector=embeddings[i],
                    payload=payload
                ))

            # 3. Upsert to Qdrant
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )
            
            logger.info(f"✅ Indexed {len(points)} chunks into Qdrant.")
            return True

        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            raise e