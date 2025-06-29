#!/usr/bin/env python3
"""
OpenAI-Compatible API Wrapper for Qwen3-Embedding-0.6B
Optimized for lightweight, high-throughput embedding tasks with 1024 dimensions.
Now with async processing and in-memory caching for superior performance.
"""

import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
import struct
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import httpx
import hashlib
import json
import os
import asyncio
from cachetools import TTLCache

# Universal Configuration with Developer Recommendations
MODEL_CONFIG = {
    "model_name": "qwen3-embedding",  # Local optimized model, served by Ollama. don't add the :latest suffix
    "dimensions": 1024,
    "max_context_length": 32768,
    "temperature": 0.0,
    "supports_instructions": True,
    "supports_mrl": True,  # Matryoshka Representation Learning
    "available_dimensions": [512, 768, 1024],  # MRL supported dimensions
    "quantization": "Q_8",
    "size_mb": 600,
    "use_case": "Instruction-aware embedding with MRL support",
    "performance_improvement": "1-5% with task-specific instructions"
}

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request with Qwen3 enhancements"""
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field(default=MODEL_CONFIG["model_name"], description="Model to use")
    encoding_format: str = Field(default="float", description="Encoding format (float or base64)")
    dimensions: Optional[int] = Field(
        default=MODEL_CONFIG["dimensions"], 
        description="Output dimensions (512, 768, or 1024 for MRL support)"
    )
    instruction: Optional[str] = Field(
        default=None, 
        description="Task-specific instruction for better performance (Qwen recommendation)"
    )
    task: Optional[str] = Field(
        default="text_search", 
        description="Task type for automatic instruction selection"
    )
    user: Optional[str] = Field(default=None, description="User ID")

class EmbeddingData(BaseModel):
    """Individual embedding data"""
    object: str = "embedding"
    embedding: Union[List[float], str]  # float array or base64 string
    index: int

class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

class Qwen3_0_6B_EmbeddingAPI:
    """
    Asynchronous Qwen3-Embedding-0.6B API wrapper with optimal configurations.
    Features async request handling, parallel batch processing, and in-memory caching.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = MODEL_CONFIG["model_name"]
        self.dimensions = MODEL_CONFIG["dimensions"]
        # Use a thread-safe in-memory cache with a 10-minute TTL and max size of 10,000 items
        self.cache = TTLCache(maxsize=10000, ttl=600)
        # Use a persistent httpx client for connection pooling
        self.client = httpx.AsyncClient(timeout=30.0)

    def _get_cache_key(self, text: str, task: str, custom_instruction: Optional[str], target_dimensions: Optional[int]) -> str:
        """Generate a unique cache key for a specific request configuration."""
        # Include all relevant parameters in the hash to ensure correctness
        payload = f"{self.model_name}:{text}:{task}:{custom_instruction}:{target_dimensions}"
        return hashlib.md5(payload.encode()).hexdigest()
    
    def _prepare_text_for_embedding(self, text: str, task: str = "text_search", custom_instruction: Optional[str] = None) -> str:
        """
        Prepare text with instruction-aware formatting (Qwen developer recommendation)
        Achieves 1-5% performance improvement with task-specific instructions
        """
        # Enhanced instruction mapping based on Qwen recommendations
        instruction_map = {
            "text_search": "Represent this text for semantic search and retrieval:",
            "code_search": "Represent this code for semantic search and similarity matching:",
            "code_indexing": "Represent this code for indexing and retrieval:",
            "document_retrieval": "Represent this document for retrieval and similarity search:",
            "question_answering": "Represent this text for question-answering tasks:",
            "clustering": "Represent this text for clustering and categorization:",
            "classification": "Represent this text for classification tasks:",
            "similarity": "Represent this text for semantic similarity comparison:",
            "general": "Represent this text for semantic understanding:"
        }
        
        if MODEL_CONFIG["supports_instructions"]:
            # Use custom instruction if provided, otherwise use task-specific instruction
            instruction = custom_instruction or instruction_map.get(task, instruction_map["text_search"])
            
            # Qwen3-Embedding optimal format: Instruction + Text
            return f"{instruction}\n{text}"
        
        # Fallback for non-instruction models
        return text
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector (Ollama doesn't auto-normalize)"""
        np_embedding = np.array(embedding)
        norm = np.linalg.norm(np_embedding)
        if norm > 0:
            return (np_embedding / norm).tolist()
        return embedding
    
    def _encode_embedding_as_base64(self, embedding: List[float]) -> str:
        """Convert float array to base64 string (RooCode OpenAI-compatible format)"""
        # Convert to Float32Array (matching OpenAI's format)
        float32_array = np.array(embedding, dtype=np.float32)
        # Convert to bytes and then base64
        bytes_data = float32_array.tobytes()
        return base64.b64encode(bytes_data).decode('utf-8')
    
    def _apply_mrl_truncation(self, embedding: List[float], target_dimensions: int) -> List[float]:
        """
        Apply MRL (Matryoshka Representation Learning) truncation to custom dimensions
        Qwen3-Embedding supports 512, 768, and 1024 dimensions
        """
        if target_dimensions not in MODEL_CONFIG["available_dimensions"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported dimension {target_dimensions}. "
                       f"Supported dimensions: {MODEL_CONFIG['available_dimensions']}"
            )
        
        if target_dimensions >= len(embedding):
            return embedding  # No truncation needed
        
        # Truncate to target dimensions (MRL property)
        truncated = embedding[:target_dimensions]
        
        # Renormalize the truncated embedding
        np_embedding = np.array(truncated)
        norm = np.linalg.norm(np_embedding)
        if norm > 0:
            return (np_embedding / norm).tolist()
        return truncated

    async def _generate_single_embedding(
        self, 
        text: str, 
        task: str,
        custom_instruction: Optional[str],
        target_dimensions: Optional[int]
    ) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        # Check cache first
        cache_key = self._get_cache_key(text, task, custom_instruction, target_dimensions)
        cached_embedding = self.cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Prepare text with instruction-aware formatting (1-5% improvement)
        formatted_text = self._prepare_text_for_embedding(text, task, custom_instruction)
        
        # Generate embedding via Ollama asynchronously
        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": formatted_text
                }
            )
            
            response.raise_for_status() # Raise exception for 4xx or 5xx status codes
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            # Validate dimensions
            if len(embedding) != self.dimensions:
                raise HTTPException(
                    status_code=500,
                    detail=f"Expected {self.dimensions} dimensions, got {len(embedding)}"
                )
            
            # Normalize embedding
            normalized_embedding = self._normalize_embedding(embedding)
            
            # Apply MRL truncation if requested (Qwen developer feature)
            if target_dimensions and MODEL_CONFIG["supports_mrl"]:
                normalized_embedding = self._apply_mrl_truncation(normalized_embedding, target_dimensions)
            
            # Cache result
            self.cache[cache_key] = normalized_embedding
            
            return normalized_embedding
            
        except httpx.HTTPStatusError as e:
            # Parse Ollama error response more gracefully
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", e.response.text)
            except json.JSONDecodeError:
                error_msg = e.response.text
            
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {error_msg}"
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        encoding_format: str = "float",
        task: str = "text_search",
        custom_instruction: Optional[str] = None,
        target_dimensions: Optional[int] = None
    ) -> EmbeddingResponse:
        """
        Generate embeddings asynchronously with parallel processing for batches.
        Uses Qwen developer recommendations (instruction-aware + MRL).
        """
        total_tokens = sum(len(text) // 4 for text in texts)
        
        # Create a list of coroutines to be executed in parallel
        tasks = [
            self._generate_single_embedding(
                text=text,
                task=task,
                custom_instruction=custom_instruction,
                target_dimensions=target_dimensions
            )
            for text in texts
        ]
        
        # Execute all embedding tasks concurrently
        try:
            float_embeddings = await asyncio.gather(*tasks)
        except Exception as e:
             # Re-raise exceptions from _generate_single_embedding as they are already HTTPException
            if isinstance(e, HTTPException):
                raise e
            # Wrap other unexpected errors in a generic 500 error
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred during batch processing: {str(e)}"
            )

        # Process results
        embedding_data_list = []
        for i, float_embedding in enumerate(float_embeddings):
            if encoding_format == "base64":
                # RooCode expects base64-encoded Float32Array
                embedding_value = self._encode_embedding_as_base64(float_embedding)
            else:
                embedding_value = float_embedding
            
            embedding_data_list.append(EmbeddingData(
                embedding=embedding_value,
                index=i
            ))
        
        return EmbeddingResponse(
            data=embedding_data_list,
            model=self.model_name,
            usage=Usage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

# FastAPI app setup
app = FastAPI(
    title="qwen3",
    description="OpenAI-compatible API for Qwen3-Embedding-0.6B model, now with async processing.",
    version="1.1.0"
)

# Initialize API with a persistent client
embedding_api = Qwen3_0_6B_EmbeddingAPI()

@app.on_event("shutdown")
async def app_shutdown():
    """Close the httpx client gracefully on shutdown."""
    await embedding_api.client.aclose()

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Qwen3-Embedding-0.6B API - RooCode Compatible",
        "model": MODEL_CONFIG["model_name"],
        "dimensions": MODEL_CONFIG["dimensions"],
        "max_context": MODEL_CONFIG["max_context_length"],
        "use_case": MODEL_CONFIG["use_case"],
        "features": [
            "Async processing with parallel batch support",
            "In-memory caching (10-minute TTL)",
            "OpenAI-compatible /v1/embeddings endpoint",
            "Base64 encoding support (encoding_format: base64)",
            "Float array support (encoding_format: float)", 
            "Normalized embeddings",
            "Instruction formatting for Qwen3",
            "Caching support"
        ],
        "endpoints": ["/v1/embeddings", "/embeddings", "/v1/models", "/health"],
        "roo_code_compatible": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test connection to Ollama
        response = await embedding_api.client.get(f"{embedding_api.ollama_url}/api/tags")
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_available = any(m["name"] == embedding_api.model_name for m in models)
        
        return {
            "status": "healthy",
            "model_available": model_available,
            "model_name": embedding_api.model_name,
            "dimensions": embedding_api.dimensions,
            "timestamp": time.time()
        }
    except httpx.RequestError as e:
        return {"status": "unhealthy", "reason": f"Ollama connection failed: {e}"}
    except httpx.HTTPStatusError as e:
        return {"status": "unhealthy", "reason": f"Ollama API error: {e.response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings (OpenAI-compatible endpoint)"""
    # Normalize input to list
    texts = [request.input] if isinstance(request.input, str) else request.input
    
    # Validate input
    if not texts or any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="Input cannot be empty or contain empty strings")
    
    if len(texts) > 256:  # Increased reasonable batch limit for async processing
        raise HTTPException(status_code=400, detail="Too many texts in batch (max 256)")
    
    # Validate encoding format
    if request.encoding_format not in ["float", "base64"]:
        raise HTTPException(
            status_code=400, 
            detail="encoding_format must be 'float' or 'base64'"
        )
    
    # Validate dimensions for MRL support
    if request.dimensions and request.dimensions not in MODEL_CONFIG["available_dimensions"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dimension {request.dimensions}. "
                   f"Supported: {MODEL_CONFIG['available_dimensions']}"
        )
    
    # Generate embeddings with Qwen developer recommendations
    return await embedding_api.generate_embeddings(
        texts=texts,
        encoding_format=request.encoding_format,
        task=request.task or "text_search",
        custom_instruction=request.instruction,
        target_dimensions=request.dimensions
    )

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_legacy(request: EmbeddingRequest):
    """Legacy embeddings endpoint for compatibility"""
    return await create_embeddings(request)

@app.get("/v1/models")
async def list_models():
    """List available models with Qwen3 feature information"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_CONFIG["model_name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen",
                "permission": [],
                "root": MODEL_CONFIG["model_name"],
                "parent": None,
                "qwen_features": {
                    "instruction_aware": MODEL_CONFIG["supports_instructions"],
                    "mrl_support": MODEL_CONFIG["supports_mrl"],
                    "available_dimensions": MODEL_CONFIG["available_dimensions"],
                    "performance_improvement": MODEL_CONFIG["performance_improvement"],
                    "recommended_tasks": [
                        "text_search", "code_search", "document_retrieval",
                        "question_answering", "clustering", "classification"
                    ]
                }
            }
        ]
    }

if __name__ == "__main__":
    print(f"🚀 Starting ASYNC Qwen3-Embedding-0.6B API server...")
    print(f"📊 Model: {MODEL_CONFIG['model_name']}")
    print(f"📏 Dimensions: {MODEL_CONFIG['dimensions']}")
    print(f"🎯 Use case: {MODEL_CONFIG['use_case']}")
    print(f"🔗 OpenAI-compatible endpoints:")
    print(f"   POST /v1/embeddings")
    print(f"   GET /v1/models")
    print(f"   GET /health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", timeout_keep_alive=60)
