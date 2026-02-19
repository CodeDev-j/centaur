"""
Retrieval Pipeline: The Read Path.

Orchestrates: query expansion → dense embedding → hybrid search → rerank → results.

Flow:
1. TermInjector expands query (multilingual synonyms + label matches)
2. Cohere embeds the ORIGINAL query (not expanded — expanded is for BM25 only)
3. Qdrant hybrid search: dense prefetch + sparse BM25 prefetch → RRF fusion
4. Voyage Rerank 2.5 re-scores the fused results
5. Score threshold filter → RetrievalResult
"""

import logging
import time
from typing import List, Optional

import voyageai
from qdrant_client import models
from langsmith import traceable

from src.config import SystemConfig
from src.schemas.retrieval import RetrievedChunk, RetrievalResult
from src.storage.vector_driver import VectorDriver
from src.retrieval.term_injector import TermInjector

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Full retrieval pipeline: expand → embed → search → rerank → filter.
    """

    def __init__(self, vector_driver: VectorDriver, term_injector: TermInjector):
        self.vector_driver = vector_driver
        self.term_injector = term_injector

        if SystemConfig.VOYAGE_API_KEY:
            self.voyage_client = voyageai.Client(api_key=SystemConfig.VOYAGE_API_KEY)
        else:
            self.voyage_client = None
            logger.warning("No VOYAGE_API_KEY — reranking disabled.")

    @traceable(name="Rerank (Voyage)", run_type="tool")
    def _rerank(
        self,
        query: str,
        documents: List[str],
        doc_indices: List[int],
        top_n: int = 10,
    ) -> List[int]:
        """
        Reranks documents using Voyage Rerank 2.5.
        Returns indices of top_n documents in relevance order.
        """
        if not self.voyage_client or not documents:
            return doc_indices[:top_n]

        try:
            result = self.voyage_client.rerank(
                query=query,
                documents=documents,
                model=SystemConfig.RERANK_MODEL,
                top_k=top_n,
            )
            return [doc_indices[r.index] for r in result.results]
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return doc_indices[:top_n]

    @traceable(name="Retrieve Context", run_type="tool")
    async def retrieve(
        self,
        query: str,
        locale: str = "en",
        top_k: int = 10,
        prefetch_k: int = 40,
        score_threshold: float = 0.0,
        filters: Optional[models.Filter] = None,
    ) -> RetrievalResult:
        """
        Full retrieval cycle:
        1. Expand query (multilingual BM25 fix)
        2. Embed ORIGINAL query via Cohere (dense vector)
        3. Hybrid search in Qdrant (dense + sparse RRF)
        4. Rerank top results via Voyage
        5. Return typed RetrievalResult
        """
        t0 = time.time()

        # 1. Query expansion (for BM25 sparse leg)
        expanded_query = await self.term_injector.expand(query, locale)

        # 2. Embed original query (for dense leg)
        dense_embeddings = await self.vector_driver._embed_texts(
            [query], input_type="search_query"
        )
        query_dense = dense_embeddings[0]

        # 3. Hybrid search: use expanded query for BM25, original embedding for dense
        raw_points = await self.vector_driver.search_hybrid(
            query_text=expanded_query,
            query_dense=query_dense,
            limit=prefetch_k,
            score_threshold=score_threshold,
            filters=filters,
        )

        if not raw_points:
            return RetrievalResult(
                chunks=[],
                query_original=query,
                query_expanded=expanded_query,
                search_latency_ms=(time.time() - t0) * 1000,
            )

        # 4. Rerank
        texts = [p.payload.get("chunk_text", "") for p in raw_points]
        indices = list(range(len(raw_points)))
        reranked_indices = self._rerank(query, texts, indices, top_n=top_k)

        # 5. Build typed results
        chunks = []
        for idx in reranked_indices:
            point = raw_points[idx]
            payload = point.payload or {}
            chunks.append(RetrievedChunk(
                chunk_id=str(point.id),
                text=payload.get("chunk_text", ""),
                score=point.score,
                page_number=int(payload.get("page_number", 0)),
                source_file=payload.get("source_file", ""),
                item_id=payload.get("item_id", ""),
                item_type=payload.get("item_type", ""),
                doc_hash=payload.get("doc_hash", ""),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in {"chunk_text", "page_number", "source_file",
                                 "item_id", "item_type", "doc_hash", "doc_id"}
                },
            ))

        latency = (time.time() - t0) * 1000
        logger.info(
            f"Retrieved {len(chunks)} chunks in {latency:.0f}ms "
            f"(prefetch={len(raw_points)}, reranked={len(reranked_indices)})"
        )

        return RetrievalResult(
            chunks=chunks,
            query_original=query,
            query_expanded=expanded_query,
            search_latency_ms=latency,
        )
