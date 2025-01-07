from src.rag.builder import RAGBuilder
from src.rag.retriever.embeddings import EmbeddingsFactory
from src.rag.generator.auth import GigaChatAuth
from src.misc.file import load_txt
from src.config import settings


template = load_txt(settings.giga_chat.prompt)
rag_builder = (
    RAGBuilder()
    .set_retriever(EmbeddingsFactory())
    .set_llm_chain(GigaChatAuth(), template)
)