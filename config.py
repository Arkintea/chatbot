# config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BACKEND_RAG_SOURCE_DIR = os.path.join(BASE_DIR, "knowledge_base")
BACKEND_RAG_INDEX_DIR = os.path.join(DATA_DIR, "backend_rag_index")
HISTORY_INDEX_DIR = os.path.join(DATA_DIR, "chat_history_index")
SESSIONS_METADATA_FILE = os.path.join(DATA_DIR, "sessions_metadata.json")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKEND_RAG_SOURCE_DIR, exist_ok=True)
os.makedirs(BACKEND_RAG_INDEX_DIR, exist_ok=True)
os.makedirs(HISTORY_INDEX_DIR, exist_ok=True)

# --- Ollama Model Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large:latest")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:14b")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))

# --- PostgreSQL Configuration ---
POSTGRES_HOST = os.getenv("PG_HOST", "raspberrypi")
POSTGRES_PORT = int(os.getenv("PG_PORT", "5432"))
POSTGRES_DATABASE = os.getenv("PG_DATABASE", "")
POSTGRES_USER = os.getenv("PG_USER", "")
POSTGRES_PASSWORD = os.getenv("PG_PASSWORD", "")
POSTGRES_CONFIGURED = os.getenv("ENABLE_POSTGRES", "False").lower() == "true"

# --- InfluxDB Configuration ---
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://raspberrypi:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "")
INFLUXDB_CONFIGURED = os.getenv("ENABLE_INFLUXDB", "False").lower() == "true"

# --- Chat Specific Constants ---
DEFAULT_CHAT_TITLE = "New Chat"