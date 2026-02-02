"""
Centralized configuration management for the Chiron Financial Forensic Pipeline.
Handles environment variables, infrastructure paths, and LLM backend instantiation
(seamlessly switching between Local/Standard OpenAI and Azure OpenAI).
"""

import os
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Suppress Pydantic V2 "protected namespace" warnings.
warnings.filterwarnings("ignore", message=".*protected namespace.*", category=UserWarning)

# Conditional imports to prevent crashes if dependencies aren't installed yet
try:
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
except ImportError:
    ChatOpenAI = None
    AzureChatOpenAI = None

# Load environment variables from .env file
load_dotenv()


class SystemConfig:
    """
    Central configuration for the Chiron engine.
    Fetches secrets from .env to prevent hardcoding credentials.
    """
    # --- DEPLOYMENT SETTINGS ---
    # Options: "LOCAL", "AZURE_PROD"
    DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "LOCAL")

    # --- MODEL NAMES ---
    # Fetches specific model identifiers from .env, with defaults
    LAYOUT_MODEL = os.getenv("LAYOUT_MODEL_NAME", "gpt-4.1-mini")
    VISION_MODEL = os.getenv("VISION_MODEL_NAME", "gpt-4.1-mini")
    REASONING_MODEL = os.getenv("REASONING_MODEL_NAME", "gpt-4.1")
    EMBEDDING_MODEL = "fastembed"  # Explicit definition

    # --- AZURE CREDENTIALS (Loaded only if needed) ---
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    # --- INFRASTRUCTURE (Required for Database connection) ---
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
        timeout: int = 300
    ):
        """
        Factory method to instantiate the correct LLM backend.
        Switches seamlessly between Standard OpenAI and Azure OpenAI based on DEPLOYMENT_MODE.

        Args:
            model_name (str): The name of the model (or Azure deployment name).
            temperature (float): Sampling temperature.
            max_tokens (int, optional): Max tokens to generate.
            timeout (int): Request timeout in seconds.

        Returns:
            ChatOpenAI | AzureChatOpenAI: Configured LLM instance.
        """
        if cls.DEPLOYMENT_MODE == "AZURE_PROD":
            if not cls.AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_PROD mode requires AZURE_OPENAI_ENDPOINT in .env")

            if AzureChatOpenAI is None:
                raise ImportError("langchain_openai not installed.")

            return AzureChatOpenAI(
                azure_deployment=model_name,  # In Azure, model_name maps to deployment_name
                api_version=cls.AZURE_OPENAI_API_VERSION,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=timeout,
                api_key=cls.AZURE_OPENAI_API_KEY,
                azure_endpoint=cls.AZURE_OPENAI_ENDPOINT
            )

        # Default: Local / Standard OpenAI
        if ChatOpenAI is None:
            raise ImportError("langchain_openai not installed.")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout
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
    LAYOUTS = BLOBS / "layouts"
    TABLES = BLOBS / "tables"
    DEFINITIONS = BLOBS / "definitions"

    # Visual Cache
    SHADOW_CACHE = DATA / "shadow_cache"

    @classmethod
    def verify(cls):
        """
        Ensures critical directories exist.
        """
        paths = [
            cls.DATA, cls.INPUTS, cls.SYSTEM,
            cls.BLOBS, cls.LAYOUTS, cls.TABLES, cls.DEFINITIONS,
            cls.SHADOW_CACHE
        ]

        for path in paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)


# Auto-verify on import
SystemPaths.verify()