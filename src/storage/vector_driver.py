"""
Vector Driver: The Search Truth (Upgraded).

Manages hybrid vector indexing and search in Qdrant:
- Dense vectors via Cohere embed-v4 (multilingual, 1024 dims)
- Sparse vectors via Qdrant native BM25 (exact term matching)
- Reciprocal Rank Fusion (RRF) for combining results

Replaces the original fastembed/bge-small-en-v1.5 driver.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import uuid4

import cohere
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from langsmith import traceable

from src.config import SystemConfig

logger = logging.getLogger(__name__)


class IndexableChunk:
    """
    Lightweight container for a chunk ready to be indexed.
    Created by src/ingestion/chunker.py, consumed here.
    """

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        doc_hash: str,
        source_file: str,
        page_number: int,
        item_id: str,
        item_type: str,
        text: str,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_hash = doc_hash
        self.source_file = source_file
        self.page_number = page_number
        self.item_id = item_id
        self.item_type = item_type
        self.text = text
        self.metadata = metadata


class VectorDriver:
    """
    Helix C: The Indexer (Upgraded).
    1. Embeds text via Cohere embed-v4 API (multilingual, 1024 dims).
    2. Generates BM25 sparse vectors via Qdrant's native tokenizer.
    3. Upserts hybrid vectors (dense + sparse) to Qdrant.
    4. Searches with Reciprocal Rank Fusion (RRF).
    """

    COLLECTION_NAME = "chiron_knowledge_base"
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)

        if not SystemConfig.COHERE_API_KEY:
            raise ValueError(
                "Missing COHERE_API_KEY in environment. "
                "Set it in .env to use Cohere embed-v4."
            )
        self.cohere_client = cohere.AsyncClientV2(
            api_key=SystemConfig.COHERE_API_KEY
        )

        # BM25 sparse embeddings via fastembed (local, no API calls)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        self._ensure_collection()
        self._ensure_payload_indices()
        logger.info(
            f"Search Truth initialized. "
            f"Collection: {self.COLLECTION_NAME} | "
            f"Dense: Cohere {SystemConfig.EMBEDDING_MODEL} ({SystemConfig.EMBEDDING_DIMS}d) | "
            f"Sparse: Qdrant BM25"
        )

    def _ensure_collection(self):
        """
        Creates collection with named dense + sparse vectors.
        Idempotent — skips if collection already exists with correct config.
        """
        if self.client.collection_exists(self.COLLECTION_NAME):
            return

        logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME}")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config={
                self.DENSE_VECTOR_NAME: models.VectorParams(
                    size=SystemConfig.EMBEDDING_DIMS,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def _ensure_payload_indices(self):
        """
        Creates payload indices for efficient scroll/count filtering.
        Idempotent — Qdrant ignores if indices already exist.
        """
        index_specs = [
            ("doc_hash", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("item_type", models.PayloadSchemaType.KEYWORD),
        ]
        for field, schema in index_specs:
            try:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass  # Index already exists

    @traceable(name="Embed Texts (Cohere)", run_type="tool")
    async def _embed_texts(
        self, texts: List[str], input_type: str = "search_document"
    ) -> List[List[float]]:
        """
        Batch embed via Cohere embed-v4.

        input_type: "search_document" for indexing, "search_query" for queries.
        """
        response = await self.cohere_client.embed(
            texts=texts,
            model=SystemConfig.EMBEDDING_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
        return response.embeddings.float_

    @traceable(name="Index Chunks (Hybrid)", run_type="tool")
    async def index(self, chunks: List[IndexableChunk]) -> bool:
        """
        Indexes chunks with both dense (Cohere) and sparse (BM25) vectors.
        """
        if not chunks:
            return False

        texts = [c.text for c in chunks]

        # 1. Dense embeddings via Cohere
        dense_embeddings = await self._embed_texts(texts, input_type="search_document")

        # 2. Sparse embeddings via fastembed BM25 (local)
        sparse_embeddings = list(self.sparse_model.embed(texts))

        # 3. Build points with named vectors and payloads
        points = []
        for i, chunk in enumerate(chunks):
            payload = {
                "doc_id": chunk.doc_id,
                "doc_hash": chunk.doc_hash,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "item_id": chunk.item_id,
                "item_type": chunk.item_type,
                "chunk_text": chunk.text,
                **chunk.metadata,
            }

            sparse = sparse_embeddings[i]
            point = models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    self.DENSE_VECTOR_NAME: dense_embeddings[i],
                    self.SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload=payload,
            )
            points.append(point)

        # 3. Upsert to Qdrant
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

        # 4. Update BM25 sparse index via Qdrant's document API
        # Qdrant native BM25 uses the payload text field for tokenization.
        # We configure a text index on chunk_text for sparse search.
        self._ensure_text_index()

        logger.info(f"Indexed {len(points)} chunks (dense + sparse).")
        return True

    def _ensure_text_index(self):
        """
        Creates a full-text index on chunk_text for Qdrant native BM25.
        Idempotent — Qdrant ignores if index already exists.
        """
        try:
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="chunk_text",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=40,
                ),
            )
        except Exception:
            pass  # Index already exists

    @traceable(name="Search Hybrid (RRF)", run_type="tool")
    async def search_hybrid(
        self,
        query_text: str,
        query_dense: Optional[List[float]] = None,
        limit: int = 20,
        score_threshold: float = 0.0,
        filters: Optional[models.Filter] = None,
    ) -> List[models.ScoredPoint]:
        """
        Executes hybrid search: dense (Cohere) + sparse (BM25), fused via RRF.

        If query_dense is not provided, embeds query_text via Cohere first.
        """
        if query_dense is None:
            embeddings = await self._embed_texts(
                [query_text], input_type="search_query"
            )
            query_dense = embeddings[0]

        # Generate sparse query vector via fastembed BM25
        sparse_results = list(self.sparse_model.embed([query_text]))
        query_sparse = models.SparseVector(
            indices=sparse_results[0].indices.tolist(),
            values=sparse_results[0].values.tolist(),
        )

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using=self.DENSE_VECTOR_NAME,
                    limit=limit,
                    filter=filters,
                ),
                models.Prefetch(
                    query=query_sparse,
                    using=self.SPARSE_VECTOR_NAME,
                    limit=limit,
                    filter=filters,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return results.points

    async def delete_by_doc_hash(self, doc_hash: str) -> None:
        """Removes all points for a given document hash."""
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_hash",
                            match=models.MatchValue(value=doc_hash),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Deleted vectors for doc_hash: {doc_hash[:8]}...")

    def scroll_by_page(self, doc_hash: str, page_number: int) -> list:
        """
        Returns all Qdrant points for a specific document page.
        Paginates via next_page_offset to never silently drop chunks.
        Full payload is returned (caller strips value_bboxes for API response).
        """
        all_points = []
        offset = None
        page_filter = models.Filter(must=[
            models.FieldCondition(
                key="doc_hash",
                match=models.MatchValue(value=doc_hash),
            ),
            models.FieldCondition(
                key="page_number",
                match=models.MatchValue(value=page_number),
            ),
        ])

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=page_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        return all_points

    def count_by_doc(self, doc_hash: str) -> Dict[str, int]:
        """
        Count chunks grouped by item_type for a document.
        Uses Qdrant's Count API with indexed filters — sub-1ms per type.
        """
        item_types = [
            "financial_table", "chart_table", "visual", "narrative", "header"
        ]
        doc_condition = models.FieldCondition(
            key="doc_hash",
            match=models.MatchValue(value=doc_hash),
        )
        counts = {}
        for item_type in item_types:
            result = self.client.count(
                collection_name=self.COLLECTION_NAME,
                count_filter=models.Filter(must=[
                    doc_condition,
                    models.FieldCondition(
                        key="item_type",
                        match=models.MatchValue(value=item_type),
                    ),
                ]),
                exact=True,
            )
            if result.count > 0:
                counts[item_type] = result.count
        return counts
