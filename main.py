from pydantic import BaseModel
from fastapi import FastAPI

from embedding.bge_m3_embedding import EmbeddingResult
from embedding.bge_m3_embedding import BgeM3EmbeddingService
from milvus.milvus_client import MilvusService

embedding_service = BgeM3EmbeddingService()
milvus_service = MilvusService(embedding_service)
app = FastAPI()

class EmbeddingRequest(BaseModel):
    texts: list[str]
    is_save: bool = True

@app.post("/hybrid-embed")
async def embed(request: EmbeddingRequest) -> EmbeddingResult:
    embedding_result = embedding_service.embed_texts(request.texts)
    if request.is_save:
        milvus_service.insert(request.texts, embedding_result)
    return embedding_result

@app.get("/search")
async def search(query: str, limit: int = 10):
    return milvus_service.hybrid_search(query, 1.0, 0.7, limit)
