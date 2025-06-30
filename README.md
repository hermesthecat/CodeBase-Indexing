# RooCode Qwen3 Codebase Indexing Setup

Qwen3-embedding has topped embedding benchmarks, easily beating both open and close-source models. This project provides tools to **optimize any Qwen3-Embedding GGUF model** downloaded through Ollama, with an OpenAI-compatible API wrapper and optimized Qdrant vector store.

**🎯 Fully RooCode Compatible!** - Works seamlessly with [Cline](https://github.com/cline/cline) and its tributaries including Roo, KiloCode.

## forked from https://github.com/OJamals/Modal

## Automated Setup (Recommended)

The entire environment can be managed using Docker Compose, which is the simplest and recommended method.

1. **Configure Your Environment:**
    - If it doesn't exist, a `.env` file will be created from the `.env.example` template when you first run the setup.
    - Review and edit the `.env` file to customize ports, API keys, or model names.

2. **Download and Optimize the AI Model:**
    - The AI model runs on your local machine using Ollama, not inside Docker. This setup provides better performance, especially on macOS.
    - Run the `optimize_gguf.py` script once to download and prepare the model for the API.

    ```bash
    # Pull the recommended base model from Ollama Hub
    ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
    
    # Run the optimizer script to create the local model file
    # This creates the .gguf file that the API service in Docker will use
    python optimize_gguf.py hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 qwen3-embedding
    ```

3. **Launch Services with Docker Compose:**
    - This single command will build the API container, start the Qdrant database, and connect everything.

    ```bash
    docker-compose up --build
    ```

    - To run in the background, add the `-d` flag: `docker-compose up --build -d`.

4. **Initialize the Vector Database:**
    - With the services running, run the `qdrantsetup.py` script once to create and configure the collection.

    ```bash
    python qdrantsetup.py
    ```

5. **(Optional) Run End-to-End Tests:**
    - Verify that everything is working correctly.

    ```bash
    python test_setup.py
    ```

**To Stop Services:**

```bash
docker-compose down
```

This setup provides a complete, optimized embedding pipeline with **Qwen developer recommendations**:

- **GGUF Model Optimizer**: `optimize_gguf.py` - Extracts and optimizes any Qwen3 model from Ollama
- **Instruction-Aware Embedding**: Task-specific instructions for 1-5% performance improvement  
- **MRL Support**: Matryoshka Representation Learning with 512, 768, and 1024 dimensions
- **OpenAI-Compatible API**: `qwen3-api.py` wrapper with RooCode base64 encoding support
- **Optimized Qdrant Vector Store**: `qdrantsetup.py` with performance tuning for 1024-dimensional vectors
- **Task-Specific Templates**: Code search, document retrieval, Q&A, clustering, and more
- **Complete RooCode Integration**: Ready-to-use with proper API keys and endpoints

## Manual Setup (Legacy)

The original setup scripts (`setup.sh`, `setup.ps1`) are still available for those who prefer not to use Docker Compose. They handle the same steps of downloading the model, starting a standalone Qdrant container, and running the Python services directly.

```bash
# For Linux/macOS
./setup.sh

# For Windows (in PowerShell)
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./setup.ps1
```

## Services

The services will run on the ports defined in your `.env` file. Defaults are:

- **Qwen3-0.6B API**: `http://localhost:8000` (Custom FastAPI wrapper, RooCode Compatible)
- **Qdrant Vector DB**: `http://localhost:6333` (Docker container with optimizations)
- **Ollama**: `http://localhost:11434` (Serving optimized GGUF model)

## RooCode Integration

After running the setup script, you'll see the exact configuration values needed for RooCode integration, based on your `.env` file:

```yaml
# RooCode Configuration (values from your .env file)
Embeddings Provider: OpenAI-compatible
Base URL: http://localhost:8000
API Key: your-super-secret-qdrant-api-key
Model: qwen3-embedding
Embedding Dimension: 1024 # 4B upto 2560; 8B upto 4096

# Vector Database Configuration
Qdrant URL: http://localhost:6333
Qdrant API Key: your-super-secret-qdrant-api-key
Collection Name: qwen3_embedding
```

Simply copy these values into your RooCode settings.

## Usage Examples

### OpenAI-Compatible API with Qwen Features

```python
import requests

# Basic embedding (uses default "text_search" task)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Your text to embed",
    "model": "qwen3",
    "encoding_format": "float"
})

# Task-specific embedding (1-5% performance improvement)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "model": "qwen3",
    "task": "code_search",  # Optimized for code
    "encoding_format": "float"
})

# Custom instruction embedding (maximum performance)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Advanced machine learning concepts",
    "model": "qwen3", 
    "instruction": "Represent this text for academic research and similarity:",
    "encoding_format": "float"
})

# MRL - Custom dimensions (Matryoshka Representation Learning)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Text for lower-dimensional embedding", 
    "model": "qwen3",
    "dimensions": 768,  # Instead of default 1024
    "encoding_format": "float"
})

embeddings = response.json()["data"][0]["embedding"]
print(f"Generated {len(embeddings)}-dimensional embedding")
```

### Task-Specific Instructions (Qwen Recommendation)

```python
# Available tasks with automatic instruction formatting:
tasks = [
    "text_search",      # General semantic search (default)
    "code_search",      # Code and programming tasks  
    "document_retrieval", # Document and text retrieval
    "question_answering", # Q&A systems
    "clustering",       # Text clustering and categorization
    "classification",   # Classification tasks
    "similarity",       # Semantic similarity comparison
    "general"          # General purpose embedding
]

# Each task automatically applies the optimal instruction format
# for 1-5% performance improvement as recommended by Qwen developers
```

### Vector Storage with Optimized Qdrant

```python
from qdrantsetup import OptimizedQdrantVectorStore

# Initialize with your settings
vs = OptimizedQdrantVectorStore(
    qdrant_url="http://localhost:6333",
    qdrant_api_key="your-super-secret-qdrant-api-key",
    embedding_api_url="http://localhost:8000",
    collection_name="qwen3_embedding"
)

# Add documents (uses optimized batching)
vs.add_document("Python is a programming language", {"category": "tech"})

# Search with filtering
results = vs.search("What is Python?", filters={"category": "tech"})
```

## Verification & Testing

```bash
# Test individual components if needed
curl http://localhost:8000/health      # API health check
curl http://localhost:6333/health      # Qdrant health check
curl http://localhost:11434/api/tags   # List Ollama models
```

## Performance Features

### High-Throughput Architecture

The API and data indexing scripts are designed for maximum performance in high-concurrency environments:

🚀 **Asynchronous API & Parallel Processing**

- The FastAPI-based server (`qwen3-api.py`) is fully asynchronous.
- It can handle hundreds of concurrent requests efficiently.
- Batch requests are processed in parallel using `asyncio.gather`, dramatically reducing latency for bulk operations.

⚡ **In-Memory Caching**

- Frequently requested embeddings are cached in memory (`cachetools.TTLCache`).
- This reduces redundant calls to the Ollama model, providing millisecond response times for repeated queries.

📦 **Optimized Batch Indexing**

- The `qdrantsetup.py` script sends documents to the embedding API in optimized batches.
- This minimizes the number of HTTP requests, significantly speeding up the process of indexing large datasets.

### Qwen Developer Recommendations (Implemented)

🚀 **Instruction-Aware Embedding**  

- Automatic task-specific instruction formatting
- 9 optimized instruction templates for different use cases

🎯 **MRL (Matryoshka Representation Learning)**  

- Support for 512, 768, and 1024 dimensions
- Smaller embeddings for faster search when full precision isn't needed
- Maintains quality with reduced dimensionality

⚡ **Optimized Configuration**  

- Memory-mapped file loading for faster startup (1-3ms load time)
- Multi-threaded processing for better performance
- Optimal context window and rope frequency settings

📊 **Benchmarked Performance**  

- Q8_0 quantization: Best quality/size balance (610MB)
- 1024-dimensional embeddings with high semantic accuracy
- Fast inference optimized for codebase indexing workflows

## API Endpoints

- `POST /v1/embeddings` - Create embeddings (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /` - API information

## Troubleshooting

**Docker Compose Issues:**

- Ensure Docker Desktop is running.
- Check that the ports defined in your `.env` file are not already in use on your machine.
- View logs for a specific service: `docker-compose logs -f api` or `docker-compose logs -f qdrant`.

**Ollama model not found:**

```bash
# Check if model exists
ollama list
# If not, recreate it
ollama create qwen3-embedding -f Modelfile
```

**API connection errors:**

```bash
# Restart API if needed
pkill -f qwen3-api.py
python qwen3-api.py &
```

**Qdrant connection issues:**

```bash
# Check Qdrant container
docker ps | grep qdrant
# Restart if needed
docker restart qdrant
```

### Performance Tuning

- **Memory**: The Q8_0 model uses ~610MB
- **CPU**: Embedding generation is CPU-intensive
- **Disk**: Qdrant stores vectors on disk, ensure sufficient space
- **Batch Size**: For large codebases, process in batches of 100-500 files

### Logs and Debugging

```bash
# Check API logs
docker-compose logs -f api

# Check Qdrant logs
docker-compose logs -f qdrant

# Check Ollama logs (if running as a service)
# systemctl status ollama
```
