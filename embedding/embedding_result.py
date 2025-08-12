from pydantic import BaseModel

class EmbeddingData(BaseModel):
    dense_vecs: list[float]
    sparse_vecs: dict[int, float]


class EmbeddingResult(BaseModel):
    embeddings: list[EmbeddingData]

    @staticmethod
    def make_bge_m3_result(text_len: int, embeddings) -> "EmbeddingResult":
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

    @staticmethod
    def make_jina_v3_result(text_len: int, embeddings) -> "EmbeddingResult":
        embedding_data_list: list[EmbeddingData] = []

        for i in range(text_len):
            embedding_data_list.append(
                EmbeddingData(
                    dense_vecs=embeddings[i].tolist(),
                    sparse_vecs=dict()  # sparse vectors are not used in Jina V3 embeddings
                )
            )

        return EmbeddingResult(embeddings=embedding_data_list)