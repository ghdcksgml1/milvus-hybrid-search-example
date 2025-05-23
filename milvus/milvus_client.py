import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker
)
from embedding.bge_m3_embedding import BgeM3EmbeddingService, EmbeddingResult

connections.connect(uri="http://home-server:19530", token="root:Milvus")

fields = [
    FieldSchema(
        name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
]
schema = CollectionSchema(fields=fields)

collection_name = "hybrid_demo"
if not utility.has_collection(collection_name):
    Collection(collection_name).drop()

col = Collection(collection_name, schema, consistency_level="Strong")

sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", index_params=sparse_index)
dense_index = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP"
}
col.create_index("dense_vector", index_params=dense_index)
col.load()


class MilvusService():
    def __init__(self, embedding_model: BgeM3EmbeddingService):
        self.client = connections
        self.docs = self._load_data()
        self.embedding_model = embedding_model

    @staticmethod
    def _load_data() -> list[str]:
        file_path = "docs/quora_duplicate_questions.tsv"
        df = pd.read_csv(file_path, sep="\t")
        questions = set()
        for _, row in df.iterrows():
            obj = row.to_dict()
            questions.add(obj["question1"][:512])
            questions.add(obj["question2"][:512])
            if len(questions) > 500:
                break
        return list(questions)

    def initialize_sample_data(self):
        for i in range(0, len(self.docs), 50):
            batch_docs = self.docs[i: i + 50]
            docs_embeddings: EmbeddingResult = self.embedding_model.embed_texts(batch_docs)

            batched_entities = []
            for j in range(len(batch_docs)):
                batched_entities.append({
                    "text": batch_docs[j],
                    "sparse_vector": docs_embeddings.embeddings[j].sparse_vecs,
                    "dense_vector": docs_embeddings.embeddings[j].dense_vecs
                })
            col.insert(batched_entities)
        print("Number of entities inserted:", col.num_entities)

    def sparse_search(self, query: str, top_k: int = 10):
        query_sparse_embedding: EmbeddingResult = self.embedding_model.embed_texts([query])
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = col.search(
            [query_sparse_embedding.embeddings[0].sparse_vecs],
            anns_field="sparse_vector",
            limit=top_k,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    def dense_search(self, query: str, top_k: int = 10):
        query_dense_embedding: EmbeddingResult = self.embedding_model.embed_texts([query])
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = col.search(
            [query_dense_embedding.embeddings[0].dense_vecs],
            anns_field="dense_vector",
            limit=top_k,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

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
        return [hit.get("text") for hit in res]