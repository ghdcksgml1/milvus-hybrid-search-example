from pydantic import BaseModel

from FlagEmbedding import BGEM3FlagModel


class EmbeddingData(BaseModel):
    dense_vecs: list[float]
    sparse_vecs: dict[int, float]


class EmbeddingResult(BaseModel):
    embeddings: list[EmbeddingData]

    @staticmethod
    def make_result(text_len: int, embeddings) -> "EmbeddingResult":
        embedding_data_list: list[EmbeddingData] = []

        for i in range(text_len):
            sparse_dict = {}
            for k, np_float in embeddings["lexical_weights"][i].items():
                sparse_dict[k] = float(np_float)

            embedding_data = EmbeddingData(
                dense_vecs=embeddings["dense_vecs"][i].tolist(),
                sparse_vecs=sparse_dict
            )
            embedding_data_list.append(embedding_data)

        return EmbeddingResult(embeddings=embedding_data_list)


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
        return EmbeddingResult.make_result(len(texts), embeddings)
