"""
Helper script to reset ChromaDB and re-ingest documents.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.chroma_db import get_db_manager
from src.database.ingestion import ingest_all_documents

if __name__ == "__main__":
    print("Resetting ChromaDB collection...")
    db = get_db_manager()
    db.reset_collection()
    print("✓ Collection reset\n")
    
    print("Starting fresh ingestion...")
    ingest_all_documents()
