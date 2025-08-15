from pydantic import BaseModel
from fastapi import FastAPI

from embedding.embedding_result import EmbeddingResult, EmbeddingData
from embedding.bge_m3_embedding import BgeM3EmbeddingService
from embedding.jina_v3_embedding import JinaV3EmbeddingService
from milvus.milvus_client import MilvusService
from milvus.milvus_client_with_jina import JinaMilvusService
from milvus.milvus_client_with_user import MilvusWithUserService

bge_m3_embedding_service = BgeM3EmbeddingService()
jina_v3_embedding_service = JinaV3EmbeddingService()

milvus_service = MilvusService(bge_m3_embedding_service)
jina_milvus_service = JinaMilvusService(jina_v3_embedding_service)
user_milvus_service = MilvusWithUserService(bge_m3_embedding_service)
app = FastAPI()


class EmbeddingRequest(BaseModel):
    texts: list[str]
    is_save: bool = True


@app.post("/hybrid-embed")
async def embed(request: EmbeddingRequest) -> EmbeddingResult:
    embedding_result = bge_m3_embedding_service.embed_texts(request.texts)
    if request.is_save:
        milvus_service.insert(request.texts, embedding_result)
    return embedding_result


@app.get("/search")
async def search(query: str, limit: int = 10):
    return milvus_service.hybrid_search(query, 1.0, 0.7, limit)


@app.post("/jina/hybrid-embed")
async def embed(request: EmbeddingRequest) -> EmbeddingResult:
    embedding_result = hybrid_embedding(request.texts)

    if request.is_save:
        jina_milvus_service.insert(request.texts, embedding_result)
    return embedding_result


@app.get("/jina/search")
async def search(query: str, limit: int = 10):
    embedding_result = hybrid_embedding([query])
    return jina_milvus_service.hybrid_search(embedding_result, 1.0, 0.7, limit)


def hybrid_embedding(texts: list[str]) -> EmbeddingResult:
    bge_m3_embedding_result = bge_m3_embedding_service.embed_texts(texts)
    jina_v3_embedding_result = jina_v3_embedding_service.embed_texts(texts)

    embedding_data_list: list[EmbeddingData] = []
    for i in range(len(texts)):
        embedding_data_list.append(
            EmbeddingData(
                dense_vecs=jina_v3_embedding_result.embeddings[i].dense_vecs,
                sparse_vecs=bge_m3_embedding_result.embeddings[i].sparse_vecs
            )
        )
    return EmbeddingResult(embeddings=embedding_data_list)


@app.post("/users/{user_id}/hybrid-embed")
async def userDataEmbed(user_id: str, request: EmbeddingRequest) -> EmbeddingResult:
    embedding_result = bge_m3_embedding_service.embed_texts(request.texts)
    if request.is_save:
        user_milvus_service.insert(user_id, request.texts, embedding_result)
    return embedding_result

@app.get("/users/{user_id}/search")
async def userDataSearch(user_id: str, query: str, limit: int = 5):
    return user_milvus_service.hybrid_search(user_id, query, 1.0, 0.7, limit)
