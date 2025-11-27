from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pathlib import Path

ENCODER_MODEL = "intfloat/multilingual-e5-base"



def chunk_text(text: str, chunk_size: int = 250, overlap: int =50, prefix="passage:") -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(prefix + text[start:end])
        start += chunk_size - overlap
    
    return chunks

with (Path(__file__).parent / "test_doc.txt").open(encoding="utf-8") as f:
    document_text = f.read()

text_chunks = chunk_text(document_text)



encoder = SentenceTransformer(ENCODER_MODEL)


client = QdrantClient(url="http://localhost:6333")


if not client.collection_exists("test_collection"):
    client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=encoder.get_sentence_embedding_dimension(), 
                                    distance=Distance.COSINE),
    )

client.upload_points(
    collection_name="test_collection",
    points=[
        PointStruct(
            id=idx,
            vector=encoder.encode(chunk).tolist(),
            payload={"text": chunk}
        )
        for idx, chunk in enumerate(text_chunks)
    ]
)

# hits = client.query_points(
#     collection_name="test_collection",
#     query=encoder.encode("Hvor gammel er Bolette?").tolist(),
#     limit=3,
# ).points

# for hit in hits:
#     print(f"Score: {hit.score:.4f}")
#     print(f"Text: {hit.payload['text']}\n")
