"""
System Configuration: The Central Nervous System
================================================

1. THE MISSION
--------------
To provide a single, immutable source of truth for application configuration.
This module bridges the gap between environmental secrets (.env), infrastructure
constants, and dynamic factory logic for LLM instantiation.

2. THE MECHANISM
----------------
- **Secret Management:** Loads environment variables via `dotenv` to prevent
  credential leakage in source control.
- **Path Abstraction:** Uses `pathlib` to create a rigorous, OS-agnostic map
  of the project directory structure (`SystemPaths`).
- **LLM Factory:** Implements the `get_llm` class method to abstract away the
  complexity of switching between Azure OpenAI and Standard OpenAI providers.

3. THE CONTRACT
---------------
- **Validation:** Critical credentials (API Keys) are checked at runtime during
  LLM initialization, raising immediate errors if missing.
- **Consistency:** All downstream tools (Layout Analyzer, Visual Extractor)
  MUST use `SystemConfig.get_llm()` to ensure uniform model parameters (temperature,
  timeouts, retries) across the pipeline.
- **Hygiene:** Automatically ensures all required data directories exist on import.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# ------------------------------------------------------------------------------
# 🔇 WARNING SUPPRESSION
# ------------------------------------------------------------------------------
# Suppress Pydantic warnings about "model_" fields in third-party libraries
# (e.g., Docling, LangChain) that conflict with Pydantic v2 protected namespaces.
warnings.filterwarnings(
    "ignore",
    message=".*Field \"model_.*\" has conflict with protected namespace.*"
)

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class SystemConfig:
    """
    Central configuration for the Chiron engine.
    Fetches secrets from .env to prevent hardcoding credentials.
    """
    DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "OPENAI_DEV")

    # --- Model Selection (role-based: change the value, not the name) ---
    # Ingestion VLM: layout detection + forensic chart extraction.
    LAYOUT_MODEL = os.getenv("LAYOUT_MODEL_NAME", "gpt-4.1-mini")
    VISION_MODEL = os.getenv("VISION_MODEL_NAME", "gpt-4.1-mini")
    # Structured reasoning: Text-to-SQL, query routing, citation verification.
    # 4.1-mini handles this well — output is constrained (SQL, classification).
    REASONING_MODEL = os.getenv("REASONING_MODEL_NAME", "gpt-4.1-mini")
    # Answer synthesis: the user-facing generation step. Needs nuance and fluency.
    GENERATION_MODEL = os.getenv("GENERATION_MODEL_NAME", "gpt-4.1")

    # --- Embedding & Reranking ---
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    # Cohere embed-v4.0: 1536 is the model's native output dimension.
    # Multilingual (100+ languages). Free tier: 1,000 calls/month.
    # Azure Marketplace deployment available for production data sovereignty.
    EMBEDDING_MODEL = "embed-v4.0"
    EMBEDDING_DIMS = 1536
    # Voyage Rerank 2.5: cross-encoder precision layer. Optional — the system
    # gracefully skips reranking if VOYAGE_API_KEY is not set.
    RERANK_MODEL = "rerank-2.5"

    # --- API Server ---
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # --- Infrastructure (REQUIRED for Docker connection) ---
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
        timeout: Optional[int] = None
    ):
        """
        Factory method to get the correct LLM without cluttering class properties.
        """
        common_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "request_timeout": timeout,
            "max_retries": 2,
        }

        # Check mode dynamically to allow runtime switching if needed
        if cls.DEPLOYMENT_MODE.upper() == "AZURE":
            return AzureChatOpenAI(
                azure_deployment=model_name,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                **common_kwargs
            )
        else:
            # Standard OpenAI (Simplest Path)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("❌ Missing OPENAI_API_KEY in environment.")

            return ChatOpenAI(
                model_name=model_name,
                api_key=api_key,
                **common_kwargs
            )


class SystemPaths:
    """
    Centralized path management for the Chiron 'Centaur' architecture.
    """
    # Root of the project
    ROOT = Path(__file__).parent.parent

    # Data Layer
    DATA = ROOT / "data"
    INPUTS = DATA / "inputs"
    SYSTEM = DATA / "system"

    # Storage Layer (Content Truth)
    BLOBS = DATA / "blobs"
    
    # Specific Artifact Containers
    TABLES = BLOBS / "tables"
    
    # [FUTURE PLACEHOLDER] For Legal Parser / Credit Agreement Analysis
    # DEFINITIONS = BLOBS / "definitions"  <-- Uncomment when Legal Parser is built

    # Visual Cache
    SHADOW_CACHE = DATA / "shadow_cache"

    @classmethod
    def verify(cls):
        """
        Ensures critical directories exist.
        """
        paths = [
            cls.DATA, cls.INPUTS, cls.SYSTEM,
            cls.BLOBS, cls.TABLES,
            cls.SHADOW_CACHE
        ]

        for path in paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)


# Auto-verify on import
SystemPaths.verify()