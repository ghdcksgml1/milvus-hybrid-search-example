from FlagEmbedding import BGEM3FlagModel

from embedding.embedding_result import EmbeddingResult

class BgeM3EmbeddingService:
    def __init__(self):
        self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_texts(self, texts: list[str], return_dense: bool = True, return_sparse: bool = True) -> EmbeddingResult:
        print(texts)
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=50,
            return_dense=return_dense,
            return_sparse=return_sparse
        )
        return EmbeddingResult.make_bge_m3_result(len(texts), embeddings)
