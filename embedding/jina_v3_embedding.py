import torch
from transformers import AutoModel

from embedding.embedding_result import EmbeddingResult

class JinaV3EmbeddingService:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3",
            trust_remote_code=True,
            device_map="auto",  # 자동 GPU 할당
            torch_dtype=torch.float16  # 메모리 절약 + 속도 향상
        ).to(device)

    def embed_texts(self, texts: list[str], return_dense: bool = True, return_sparse: bool = True) -> EmbeddingResult:
        print(texts)
        embeddings = self.embedding_model.encode(
            texts,
            task="text-matching"
        )
        return EmbeddingResult.make_jina_v3_result(len(texts), embeddings)