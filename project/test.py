
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    encode_kwargs={
        "normalize_embeddings": True,
        "truncate_dim": 128,
    },
    query_encode_kwargs={
        "normalize_embeddings": True,
        "truncate_dim": 128,
    },
)

text = "Spring Boot is a Java framework."

vector = embeddings.embed_query(text)

print("Input text:", text)
print("Embedding type:", type(vector))
print("Embedding dimensions:", len(vector))
print("First 10 values:", vector[:10])
print("Full embedding:", vector)
