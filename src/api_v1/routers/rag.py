from fastapi import (
    APIRouter,
    Query,
    Depends,
    status
)
from fastapi.responses import JSONResponse

# from src.api_v1.dependencies import get_rag_chain
from src.rag.abstract import AbstractChain
from src.schemas import AnswerResponse
from src.config import settings

from src.rag.builder import RAGBuilder
from src.rag.retriever.embeddings import EmbeddingsFactory
from src.rag.generator.auth import GigaChatAuth
from src.misc.file import load_txt


rag_router = APIRouter(
    prefix=f"{settings.api_v1.prefix}/rag",
    tags=["RAG GigaChat"]
)


@rag_router.post(path="/answer/", response_model=AnswerResponse)
async def get_rag_answer(
        query: str = Query(...),
        # rag_chain: AbstractChain = Depends(get_rag_chain)
) -> JSONResponse:
    template = await load_txt(settings.giga_chat.prompt)
    rag_builder = (
        RAGBuilder()
        .set_retriever(EmbeddingsFactory())
        .set_llm_chain(GigaChatAuth(), template)
    )
    rag_chain = rag_builder.get_rag_chain()
    answer: str = await rag_chain.invoke(query)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "data": {
                "status": "ok",
                "answer": answer
            }
        }
    )
