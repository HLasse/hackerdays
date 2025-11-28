"""Script to do to super simple RAG with Qdrant and SentenceTransformers."""

from typing import Generator, Iterator, Union
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ollama import Client as OllamaClient
import os


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "RAG Pipeline"
        qdrant_url: str
        collection_name: str
        ollama_url: str
        ollama_model: str
        top_k: int

    def __init__(self):
        self.ENCODER_MODEL = "intfloat/multilingual-e5-base"
        self.valves = self.Valves(
            **{
                "qdrant_url": os.getenv("qdrant_url", "http://localhost:6333"),
                "collection_name": os.getenv("collection_name", "test_collection"),
                "ollama_url": os.getenv("ollama_url", "http://localhost:11434"),
                "ollama_model": os.getenv("ollama_model", "qwen3:4b"),
                "top_k": int(os.getenv("top_k", 3)),
            }
        )

    def _build_clients(self):
        # self.encoder = SentenceTransformer(self.ENCODER_MODEL)
        # self.db_client = QdrantClient(url=self.valves.qdrant_url)
        # self.generator = OllamaClient(host=self.valves.ollama_url)
        pass

    async def on_startup(self):
        self._build_clients()
        print("Startup complete.")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: list[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        self.encoder = SentenceTransformer(self.ENCODER_MODEL)
        self.db_client = QdrantClient(url=self.valves.qdrant_url)
        self.generator = OllamaClient(host=self.valves.ollama_url)
        
        # RAG main prompt
        prompt = "You are a helpful AI assistant. Use the following context to answer the question."

        # convert message to vector
        query_vector = self.encoder.encode("passage: " + user_message).tolist()
        # query Qdrant
        hits = self.db_client.query_points(
            collection_name=self.valves.collection_name,
            query=query_vector,
            limit=self.valves.top_k,
        ).points

        # extract relevant documents
        relevant_docs = [hit.payload["text"] for hit in hits]
        # prepare context for LLM
        context = "\n\n".join(relevant_docs)
        # prepare prompt
        full_prompt = f"{prompt}\n\nContext:\n{context}\n\nQuestion: {user_message}"

        # add context to messages
        messages.append({"role": "system", "content": full_prompt})
        # query LLM
        response = self.generator.chat(
            model=self.valves.ollama_model, messages=messages
        )
        # can also return thinking streams if needed
        return response.message.content


# if __name__ == "__main__":
#     pipeline = Pipeline()
#     import asyncio

#     asyncio.run(pipeline.on_startup())
#     response = pipeline.pipe(
#         user_message="Hvad er Bolettes job?",
#         model_id="llama2",
#         messages=[{"role": "user", "content": "Hvad er Bolettes job?"}],
#         body={},
#     )
#     print("Response:", response)
