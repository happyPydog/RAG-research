from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
)  # vector size 768
