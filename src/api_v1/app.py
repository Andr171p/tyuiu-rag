from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api_v1.lifespan import lifespan
from src.api_v1.routers import rag_router
from src.config import settings


app = FastAPI(
    title=settings.api_v1.name,
    lifespan=lifespan
)

app.include_router(rag_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
