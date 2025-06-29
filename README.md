# RooCode Qwen3 Codebase Indexing Setup

Qwen3-embedding has topped embedding benchmarks, easily beating both open and close-source models. This project provides tools to **optimize any Qwen3-Embedding GGUF model** downloaded through Ollama, with an OpenAI-compatible API wrapper and optimized Qdrant vector store.

**🎯 Fully RooCode Compatible!** - Works seamlessly with [Cline](https://github.com/cline/cline) and its tributaries including Roo, KiloCode.

## Automated Setup (Recommended)

The setup scripts (`setup.sh` for Linux/macOS, `setup.ps1` for Windows) automate the entire process.

1. **Configure Environment:**
    - The script will automatically create a `.env` file from the `.env.example` template if it doesn't exist.
    - **Before running the script**, you can review and edit the `.env` file to customize ports, API keys, etc.

2. **Run the Script:**

    For Linux or macOS:

    ```bash
    # One-command setup: downloads model, optimizes, and configures everything
    ./setup.sh
    ```

    For Windows (in PowerShell):

    ```powershell
    # Make sure execution policy allows running scripts
    # Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\setup.ps1
    ```

This automated script:

- Downloads Qwen3-Embedding-0.6B model (Q8_0-optimized) via Ollama
- Extracts and optimizes the GGUF model from Ollama storage  
- Creates optimized Ollama model for embedding-only usage
- Installs Python dependencies from `requirements.txt`
- Starts all services (Qdrant, API) using the configuration from your `.env` file.
- Sets up the Qdrant vector database with the correct configuration.

## Manual Setup (Advanced Users)

1. **Create Configuration File:**
    - Copy `.env.example` to `.env`.
    - Edit the `.env` file with your desired settings (API keys, ports, etc.).

2. **Download and Optimize Model:**

    ```bash
    # Pull the recommended model
    ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
    
    # Run the optimizer script to create the local model
    python optimize_gguf.py hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 qwen3-embedding
    ```

3. **Install Dependencies and Start Services:**

    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # Start Qdrant using Docker (it will read the API key from your .env file)
    docker run -d --name qdrant -p 6333:6333 \
      -e QDRANT__SERVICE__API_KEY=$(grep QDRANT_API_KEY .env | cut -d '=' -f2) \
      -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    
    # Start the API (it will read settings from your .env file)
    python qwen3-api.py
    ```

4. **Setup Vector Database:**

    ```bash
    # This script also reads from .env to connect to the services
    python qdrantsetup.py
    ```

**Ready to use with RooCode!** The setup script displays the exact configuration values needed.

This setup provides a complete, optimized embedding pipeline with **Qwen developer recommendations**:

- **GGUF Model Optimizer**: `optimize_gguf.py` - Extracts and optimizes any Qwen3 model from Ollama
- **Instruction-Aware Embedding**: Task-specific instructions for 1-5% performance improvement  
- **MRL Support**: Matryoshka Representation Learning with 512, 768, and 1024 dimensions
- **OpenAI-Compatible API**: `qwen3-api.py` wrapper with RooCode base64 encoding support
- **Optimized Qdrant Vector Store**: `qdrantsetup.py` with performance tuning for 1024-dimensional vectors
- **Task-Specific Templates**: Code search, document retrieval, Q&A, clustering, and more
- **Complete RooCode Integration**: Ready-to-use with proper API keys and endpoints

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
# Check API logs (run in foreground to see live output)
python qwen3-api.py

# Check Qdrant logs
docker logs qdrant

# Check Ollama logs
ollama logs qwen3-embedding
```
