"""
System Configuration
====================
Single source of truth for secrets (.env), model selection, infrastructure
constants, and LLM factory logic. All downstream consumers MUST use
SystemConfig.get_llm() for uniform model parameters across the pipeline.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# Suppress Pydantic v2 "model_" namespace warnings from third-party libs
warnings.filterwarnings(
    "ignore",
    message=".*Field \"model_.*\" has conflict with protected namespace.*"
)

load_dotenv()
logger = logging.getLogger(__name__)


class SystemConfig:
    """Central configuration — reads from .env, provides LLM factory."""

    # === Deployment Mode ===
    DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "OPENAI_DEV")

    # === Model Selection (role-based) ===
    # Ingestion pipeline (high-volume, cost-sensitive)
    LAYOUT_MODEL = os.getenv("LAYOUT_MODEL_NAME", "gpt-4.1-mini")
    VISION_MODEL = os.getenv("VISION_MODEL_NAME", "gpt-4.1-mini")
    # Query pipeline (routing, text-to-SQL — constrained output)
    REASONING_MODEL = os.getenv("REASONING_MODEL_NAME", "gpt-4.1-mini")
    # Answer generation (user-facing, needs nuance)
    GENERATION_MODEL = os.getenv("GENERATION_MODEL_NAME", "gpt-4.1-mini")

    # === Embedding & Reranking ===
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    EMBEDDING_MODEL = "embed-v4.0"
    EMBEDDING_DIMS = 1536           # Cohere embed-v4.0 native dimension
    RERANK_MODEL = "rerank-2.5"     # Voyage cross-encoder; skipped if no API key

    # === API Server ===
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # === Infrastructure (Postgres — matches docker-compose.yml) ===
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "chiron_ledger")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    @classmethod
    def get_llm(
        cls,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """Factory: returns the correct LLM provider based on DEPLOYMENT_MODE."""
        common_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "request_timeout": timeout,
            "max_retries": 2,
        }

        if cls.DEPLOYMENT_MODE.upper() == "AZURE":
            return AzureChatOpenAI(
                azure_deployment=model_name,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                **common_kwargs,
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")

        return ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            **common_kwargs,
        )


class SystemPaths:
    """OS-agnostic directory layout. Auto-creates on import."""

    ROOT = Path(__file__).parent.parent

    # Data layer
    DATA = ROOT / "data"
    INPUTS = DATA / "inputs"
    SYSTEM = DATA / "system"

    # Blob storage
    BLOBS = DATA / "blobs"
    TABLES = BLOBS / "tables"

    # Visual cache (rendered page images)
    SHADOW_CACHE = DATA / "shadow_cache"

    @classmethod
    def verify(cls):
        """Ensures all required directories exist."""
        for path in [cls.DATA, cls.INPUTS, cls.SYSTEM, cls.BLOBS, cls.TABLES, cls.SHADOW_CACHE]:
            path.mkdir(parents=True, exist_ok=True)


# Auto-verify on import
SystemPaths.verify()
