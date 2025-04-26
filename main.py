from typing import Annotated

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from embedding.bge_m3_embedding import EmbeddingResult
from embedding.bge_m3_embedding import BgeM3EmbeddingService
from milvus.milvus_client import MilvusService

embedding_service = BgeM3EmbeddingService()
milvus_service = MilvusService(embedding_service)
app = FastAPI()

class EmbeddingRequest(BaseModel):
    texts: list[str]

@app.post("/hybrid-embed")
async def embed(request: EmbeddingRequest) -> EmbeddingResult:
    return embedding_service.embed_texts(request.texts)

@app.post("/sample-data")
async def initialize_sample_data() -> str:
    milvus_service.initialize_sample_data()
    return "Sample data initialized successfully."

@app.get("/search")
async def search(query: str, dense: bool = True, sparse: bool = False, limit: int = 10):
    if dense and sparse:
        return milvus_service.hybrid_search(query, 1.0, 0.7, limit)
    elif dense:
        return milvus_service.dense_search(query, limit)
    elif sparse:
        return milvus_service.sparse_search(query, limit)
    raise HTTPException(status_code=404, detail="Not found.")