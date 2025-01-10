from fastapi import (
    APIRouter,
    Query,
    Depends,
    status
)
from fastapi.responses import JSONResponse

from src.rag.abstract import AbstractChain
from src.api_v1.dependencies import get_rag_chain
from src.schemas import AnswerResponse
from src.config import settings


rag_router = APIRouter(
    prefix=f"{settings.api_v1.prefix}/rag",
    tags=["RAG GigaChat"]
)


@rag_router.get(path="/answer/", response_model=AnswerResponse)
async def get_rag_answer(
        query: str = Query(...),
        rag_chain: AbstractChain = Depends(get_rag_chain)
) -> JSONResponse:
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
