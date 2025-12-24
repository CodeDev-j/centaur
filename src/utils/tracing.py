import os
import logging
from langsmith import Client

logger = logging.getLogger(__name__)

def verify_tracing():
    """
    Simple check to ensure LangSmith is reachable.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning("⚠️ LANGCHAIN_API_KEY is missing. Tracing will not work.")
        return False
        
    try:
        client = Client()
        # Ping the API by listing projects (limit 1)
        projects = list(client.list_projects(limit=1))
        logger.info(f"✅ LangSmith Connected! Found {len(projects)} existing projects.")
        return True
    except Exception as e:
        logger.error(f"❌ LangSmith Connection Failed: {e}")
        return False

if __name__ == "__main__":
    # Allow running this file directly to test connections
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    verify_tracing()