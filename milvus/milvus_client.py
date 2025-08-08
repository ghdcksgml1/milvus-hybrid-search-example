from fastapi import HTTPException
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker, MilvusException
)
from embedding.bge_m3_embedding import BgeM3EmbeddingService, EmbeddingResult

connections.connect(uri="http://home-server:19530", token="root:Milvus")

fields = [
    FieldSchema(
        name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
]
schema = CollectionSchema(fields=fields)

collection_name = "bge_m3_hybrid_search"
col = Collection(collection_name, schema, using="default")
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", index_params=sparse_index)
dense_index = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP"
}
col.create_index("dense_vector", index_params=dense_index)
col.load()

class MilvusService:
    def __init__(self, embedding_model: BgeM3EmbeddingService):
        self.client = connections
        self.embedding_model = embedding_model

    def insert(self, texts: list[str], embedding_result: EmbeddingResult):
        try:
            batched_entities = []
            for i in range(len(texts)):
                batch_text = texts[i]
                batch_embedding = embedding_result.embeddings[i]

                batched_entities.append({
                    "text": batch_text,
                    "sparse_vector": batch_embedding.sparse_vecs,
                    "dense_vector": batch_embedding.dense_vecs
                })

            col.insert(batched_entities)
            print("Number of entities inserted:", batched_entities.__len__())
        except MilvusException as e:
            print(f"Error inserting data into Milvus: {e}")
            raise HTTPException(status_code=500, detail="Error inserting data into Milvus")

    def hybrid_search(self, query: str, dense_weight: float = 1.0, sparse_weight: float = 1.0, top_k: int = 10):
        query_embedding = self.embedding_model.embed_texts([query])
        dense_search_params = {
            "metric_type": "IP",
            "params": {},
        }
        dense_req = AnnSearchRequest(
            [query_embedding.embeddings[0].dense_vecs], "dense_vector", dense_search_params, limit=top_k
        )
        sparse_search_params = {
            "metric_type": "IP",
            "params": {},
        }
        sparse_req = AnnSearchRequest(
            [query_embedding.embeddings[0].sparse_vecs], "sparse_vector", sparse_search_params, limit=top_k
        )

        rerank = WeightedRanker(dense_weight, sparse_weight)
        res = col.hybrid_search(
            [dense_req, sparse_req], rerank=rerank, limit=top_k, output_fields=["text"]
        )[0]
        print(f"Hybrid search results: {res}")
        return [hit.get("text") for hit in res]