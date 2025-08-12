from transformers import AutoModel

from embedding.embedding_result import EmbeddingResult

class JinaV3EmbeddingService:
    def __init__(self):
        self.embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

    def embed_texts(self, texts: list[str], return_dense: bool = True, return_sparse: bool = True) -> EmbeddingResult:
        print(texts)
        embeddings = self.embedding_model.encode(
            texts,
            task="text-matching"
        )
        return EmbeddingResult.make_jina_v3_result(len(texts), embeddings)