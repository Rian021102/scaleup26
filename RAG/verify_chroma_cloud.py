"""Standalone check that Chroma Cloud is reachable and writable.

Usage:
    export CHROMA_API_KEY="ck-..."
    export CHROMA_TENANT="your-tenant-uuid"
    export CHROMA_DATABASE="your-database-name"
    python verify_chroma_cloud.py

Exits 0 on a successful write/read round-trip, 1 otherwise.
"""

import os
import sys
import uuid
from pathlib import Path

import chromadb
from dotenv import load_dotenv

# Load Chroma Cloud credentials from the .env file next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")


def main() -> int:
    missing = [v for v in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE") if not os.environ.get(v)]
    if missing:
        print(f"[FAIL] Missing env vars: {', '.join(missing)}")
        return 1

    print("Connecting to Chroma Cloud...")
    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )

    # heartbeat proves the network/auth path works
    client.heartbeat()
    print("[OK] heartbeat succeeded (auth + network OK)")

    # write/read round-trip in a throwaway collection
    name = f"verify{uuid.uuid4().hex[:8]}"
    col = client.get_or_create_collection(name=name)
    col.add(
        ids=["doc1"],
        documents=["hello from chroma cloud"],
        # a trivial 3-dim embedding so we don't need Ollama for this test
        embeddings=[[0.1, 0.2, 0.3]],
    )
    got = col.get(ids=["doc1"])
    assert got["documents"] == ["hello from chroma cloud"], got
    print(f"[OK] wrote and read back a document in collection '{name}' (count={col.count()})")

    client.delete_collection(name=name)
    print("[OK] cleaned up test collection")

    print("\nExisting collections in this database:")
    for c in client.list_collections():
        print(f"  - {c.name} (count={c.count()})")

    print("\n[SUCCESS] Chroma Cloud is reachable and writable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
