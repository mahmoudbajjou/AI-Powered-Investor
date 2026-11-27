import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
MARKDOWN_DIR = DATA_DIR / "markdown"
DATABASE_PATH = ROOT / os.getenv("INVESTOR_DB_PATH", "data/investor.sqlite3")

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
