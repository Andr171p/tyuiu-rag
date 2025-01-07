from contextlib import (
    asynccontextmanager,
    AbstractAsyncContextManager
)

from fastapi import FastAPI

from src.rag.builder import RAGBuilder
from src.rag.retriever.embeddings import EmbeddingsFactory
from src.rag.generator.auth import GigaChatAuth
from src.misc.file import load_txt
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AbstractAsyncContextManager[None]:
    template = await load_txt(settings.giga_chat.prompt)
    app.state.rag_builder = (
        RAGBuilder()
        .set_retriever(EmbeddingsFactory())
        .set_llm_chain(GigaChatAuth(), template)
    )
    yield
    del app.state.rag_builder
