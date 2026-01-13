import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SystemConfig:
    """
    Central configuration for the Chiron engine.
    Fetches secrets from .env to prevent hardcoding credentials.
    """
    DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "OPENAI_DEV")
    
    # Model Selection
    # Wired up to .env so you can toggle costs easily
    LAYOUT_MODEL = os.getenv("LAYOUT_MODEL_NAME", "gpt-4.1-mini") 
    VISION_MODEL = os.getenv("VISION_MODEL_NAME", "gpt-4.1")
    REASONING_MODEL = os.getenv("REASONING_MODEL_NAME", "gpt-4.1")
    EMBEDDING_MODEL = "fastembed" 
    
    # Infrastructure (REQUIRED for Docker connection)
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "chiron_ledger")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

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