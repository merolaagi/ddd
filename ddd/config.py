import os
from pathlib import Path

ROOT = Path(os.environ.get("DDD_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "results.db"
MODEL_DIR = DATA_DIR / "models"

WEB_PORT = int(os.environ.get("DDD_WEB_PORT", "47520"))
MCP_PORT = int(os.environ.get("DDD_MCP_PORT", "47521"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

for d in (DATA_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
