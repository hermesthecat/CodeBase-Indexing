#!/bin/bash
# Qwen3 Embedding Setup Script for Windows
# Automates the complete setup process for RooCode integration

# Stop on any error
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Qwen3 Embedding for RooCode on Windows..."
Write-Host "============================================="

# Check if required files exist
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ requirements.txt not found!"
    exit 1
}

# Check for .env file and create from example if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "🔍 .env file not found. Creating from .env.example..."
    if (-not (Test-Path ".env.example")) {
        Write-Host "❌ .env.example not found! Cannot create .env file."
        exit 1
    }
    Copy-Item -Path ".env.example" -Destination ".env"
    Write-Host "✅ .env file created. Please review and edit it if necessary."
}

# Load environment variables from .env file
Write-Host "📦 Loading configuration from .env file..."
try {
    Get-Content .env | ForEach-Object {
        if ($_ -match "^\s*#") { return } # Skip comments
        $parts = $_.Split("=", 2)
        if ($parts.Length -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            # Remove quotes if present
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            [System.Environment]::SetEnvironmentVariable($key, $value)
            # Also set for the current session for immediate use
            $env:$key = $value
        }
    }
} catch {
    Write-Host "❌ Failed to load environment variables from .env: $_"
    exit 1
}

# Check if ollama is available
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Ollama not found! Please install Ollama first."
    Write-Host "   You can download it from: https://ollama.com/download"
    exit 1
}

# Step 1: Download and optimize GGUF model
Write-Host "📦 Step 1: Setting up Qwen3 embedding model..."

# Check if model already exists locally
if (-not (Test-Path "qwen3-embedding.gguf")) {
    Write-Host "🔍 GGUF model not found locally. Checking Ollama models..."
    
    # Check if model exists in Ollama
    $ollama_models = ollama list
    if (-not ($ollama_models | Select-String "qwen3-embedding")) {
        Write-Host "📥 Downloading Qwen3-Embedding-0.6B model (Q8_0-optimized)..."
        ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
        
        # Wait a moment for the model to be fully registered
        Start-Sleep -Seconds 2
    }
    
    # Run GGUF optimizer to extract and optimize the model
    Write-Host "⚙️ Optimizing GGUF model from Ollama storage..."
    try {
        python optimize_gguf.py hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 qwen3-embedding
        Write-Host "✅ GGUF model optimized successfully"
    } catch {
        Write-Host "❌ Failed to optimize GGUF model: $_"
        Write-Host "Trying alternative approach..."
        
        # Alternative: create Modelfile without local GGUF
        $modelfileContent = @"
FROM hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
PARAMETER num_ctx 8192
PARAMETER embedding_only true
"@
        Set-Content -Path "Modelfile" -Value $modelfileContent
        ollama create qwen3-embedding -f Modelfile
        Write-Host "✅ Ollama model created from remote GGUF"
    }
} else {
    Write-Host "✅ Local GGUF model found: qwen3-embedding.gguf"
    
    # Create optimized Modelfile for local GGUF
    $modelfileContent = @"
FROM ./qwen3-embedding.gguf
PARAMETER num_ctx 8192
PARAMETER embedding_only true
"@
    Set-Content -Path "Modelfile" -Value $modelfileContent
    
    # Create the model in Ollama
    Write-Host "🔧 Creating optimized Ollama model..."
    ollama create qwen3-embedding -f Modelfile
    Write-Host "✅ Ollama model created successfully"
}

# Step 2: Install Python dependencies
Write-Host "📦 Step 2: Installing Python dependencies..."
pip install -r requirements.txt
Write-Host "✅ Dependencies installed"

# Step 3: Setup Qdrant
Write-Host "📦 Step 3: Setting up Qdrant vector database..."

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker is not running! Please start Docker Desktop first."
    exit 1
}

# Stop existing Qdrant container if it exists
$existingContainer = docker ps -a --format '{{.Names}}' | Select-String -Quiet "qdrant"
if ($existingContainer) {
    Write-Host "Stopping and removing existing Qdrant container..."
    docker stop qdrant
    docker rm qdrant
}

# Start new Qdrant container using API key from environment
Write-Host "Starting Qdrant container..."
# Using ${PWD} which is a cross-platform way to define the volume in PowerShell
docker run -d --name qdrant `
    -p 6333:6333 `
    -p 6334:6334 `
    -e "QDRANT__SERVICE__API_KEY=$($env:QDRANT_API_KEY)" `
    -v "${PWD}/qdrant_storage:/qdrant/storage" `
    qdrant/qdrant

# Wait for Qdrant to be ready
Write-Host "Waiting for Qdrant to start..."
for ($i=1; $i -le 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri http://localhost:6333/health -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Qdrant is ready"
            break
        }
    } catch {}
    if ($i -eq 30) {
        Write-Host "❌ Qdrant failed to start"
        exit 1
    }
    Start-Sleep -Seconds 2
}

# Step 4: Start the API in a background process
Write-Host "📦 Step 4: Starting OpenAI-compatible API..."
$apiProcess = Start-Process python -ArgumentList "qwen3-api.py" -PassThru -WindowStyle Hidden

# Wait for API to be ready
Write-Host "Waiting for API to start..."
for ($i=1; $i -le 20; $i++) {
    try {
        $response = Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ API is ready"
            break
        }
    } catch {}
    if ($i -eq 20) {
        Write-Host "❌ API failed to start"
        Stop-Process -Id $apiProcess.Id -Force
        exit 1
    }
    Start-Sleep -Seconds 2
}

# Step 5: Setup Qdrant vector store
Write-Host "📦 Step 5: Setting up Qdrant vector store..."
python qdrantsetup.py
Write-Host "✅ Qdrant vector store configured"

# Step 6: Run verification tests
Write-Host "📦 Step 6: Running verification tests..."
python test_setup.py

# Summary
Write-Host ""
Write-Host "🎉 Setup complete! Your Qwen3 embedding system is ready for RooCode."
Write-Host ""
Write-Host "🔧 RooCode Configuration:"
Write-Host "   Embeddings Provider: OpenAI-compatible"
Write-Host "   Base URL: $($env:EMBEDDING_API_URL)"
Write-Host "   API Key: $($env:QDRANT_API_KEY)"
Write-Host "   Model: qwen3-embedding"
Write-Host "   Embedding Dimension: 1024"
Write-Host "   Qdrant URL: $($env:QDRANT_URL)"
Write-Host "   Qdrant API Key: $($env:QDRANT_API_KEY)"
Write-Host "   Collection Name: qwen3_embedding"
Write-Host ""
Write-Host "🚀 Services running:"
Write-Host "   - Qwen3 API: http://localhost:8000 (Process ID: $($apiProcess.Id))"
Write-Host "   - Qdrant: http://localhost:6333 (Docker container)"
Write-Host "   - Ollama: http://localhost:11434"
Write-Host ""
Write-Host "💡 To stop the API, run: Stop-Process -Id $($apiProcess.Id)"
Write-Host "💡 To stop Qdrant, run: docker stop qdrant"
Write-Host "💡 To restart, run this script again: .\setup.ps1" 