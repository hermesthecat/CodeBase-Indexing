#!/usr/bin/env python3
"""
End-to-End Test Suite for the Qwen3 Embedding Setup

This script verifies that all services (Ollama, Qwen3 API, Qdrant) are running
and integrated correctly after the setup process.

It performs the following checks:
1. Pings the health endpoints of the API and Qdrant.
2. Creates an embedding via the API.
3. Adds the embedding to the Qdrant collection.
4. Searches for the embedding in Qdrant to verify storage and retrieval.
"""

import os
import sys
import time
import requests
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import uuid

# Load configuration from .env file
load_dotenv()

# --- Configuration ---
API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "qwen3_embedding"
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3-embedding")
EXPECTED_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 1024))

# --- Helper Functions ---
def print_test(name):
    """Prints a formatted test header."""
    print(f"\n🧪 Running test: {name}...")

def print_success(message):
    """Prints a formatted success message."""
    print(f"✅ SUCCESS: {message}")

def print_failure(message, exit_code=1):
    """Prints a formatted failure message and exits."""
    print(f"❌ FAILURE: {message}", file=sys.stderr)
    sys.exit(exit_code)

def wait_for_service(url, service_name, max_retries=15, delay=2):
    """Waits for a service to become available by pinging its health endpoint."""
    print(f"⏳ Waiting for {service_name} at {url}...")
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready.")
                return True
        except requests.exceptions.RequestException:
            pass # Ignore connection errors while waiting
        time.sleep(delay)
    print_failure(f"{service_name} did not become available after {max_retries * delay} seconds.")

# --- Test Functions ---

def test_service_health():
    """Test 1: Check if all services are healthy and responsive."""
    print_test("Service Health Check")
    wait_for_service(f"{API_URL}/health", "Qwen3 API")
    wait_for_service(f"{QDRANT_URL}/health", "Qdrant")
    print_success("All services are healthy and responsive.")

def test_embedding_creation():
    """Test 2: Create a new embedding via the API."""
    print_test("Embedding Creation")
    test_text = f"This is an end-to-end test run at {time.time()}"
    
    try:
        response = requests.post(
            f"{API_URL}/v1/embeddings",
            json={
                "input": test_text,
                "model": MODEL_NAME,
                "encoding_format": "float"
            },
            timeout=45  # Allow more time for model loading on first run
        )
        response.raise_for_status()
        data = response.json()

        # --- Verification ---
        if not ("data" in data and len(data["data"]) == 1):
            print_failure("API response should contain exactly one embedding in the 'data' list.")
        
        embedding_data = data["data"][0]
        if "embedding" not in embedding_data:
            print_failure("Embedding data not found in API response.")
            
        embedding = embedding_data["embedding"]
        if not isinstance(embedding, list):
            print_failure("Embedding should be a list of floats.")
            
        if len(embedding) != EXPECTED_DIMENSIONS:
            print_failure(f"Embedding dimensions mismatch. Expected {EXPECTED_DIMENSIONS}, got {len(embedding)}.")
        
        print_success(f"Successfully created a {len(embedding)}-dimensional embedding.")
        return test_text, embedding

    except requests.exceptions.HTTPError as e:
        print_failure(f"API returned an HTTP error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print_failure(f"Could not connect to the embedding API: {e}")
    except Exception as e:
        print_failure(f"An unexpected error occurred during embedding creation: {e}")

def test_qdrant_integration(doc_text, embedding):
    """Test 3: Verify Qdrant integration by upserting and searching for the created embedding."""
    print_test("Qdrant Integration")
    
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=20)

        # 1. Upsert the point
        point_id = str(uuid.uuid4())
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": doc_text, "source": "e2e-test"}
                )
            ],
            wait=True  # Wait for the operation to complete on the server side
        )
        print("   - Upserted test point to Qdrant collection.")

        # Give it a moment for indexing to be fully consistent, although wait=True helps.
        time.sleep(1)

        # 2. Search for the point to verify it was stored correctly
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=1,
            with_payload=True,
        )
        print("   - Searched for the test point using its embedding.")
        
        # --- Verification ---
        if not search_results:
            print_failure("Search returned no results. The point was not found in Qdrant.")
            
        top_result = search_results[0]
        if top_result.id != point_id:
            print_failure(f"Retrieved point ID '{top_result.id}' does not match the upserted ID '{point_id}'.")
            
        if top_result.score < 0.99:
            print_failure(f"Similarity score is too low ({top_result.score:.4f}). Expected > 0.99 for an exact match.")
        
        if top_result.payload.get("text") != doc_text:
            print_failure("The payload of the retrieved point does not match the original text.")

        print_success("Successfully upserted and retrieved the test point from Qdrant.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print_failure(f"An unexpected error occurred during Qdrant integration: {e}")

def main():
    """Runs the full suite of end-to-end tests."""
    print("🚀 Starting End-to-End Test Suite for Qwen3 Embedding Setup")
    print("=============================================================")
    
    try:
        test_service_health()
        test_text, embedding = test_embedding_creation()
        test_qdrant_integration(test_text, embedding)

        print("\n=============================================================")
        print("🎉 ALL TESTS PASSED! The system is fully operational.")
        print("=============================================================")
    except Exception as e:
        # This will catch failures from print_failure which calls sys.exit
        pass

if __name__ == "__main__":
    main() 