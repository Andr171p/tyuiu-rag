import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal, Dict
from pydantic_settings import BaseSettings


BASE_DIR: Path = Path(__file__).resolve().parent.parent

ENV_PATH: Path = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


class EmbeddingsSettings(BaseSettings):
    # model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_name: str = "d0rj/e5-base-en-ru"
    model_kwargs: Dict[str, str] = {"device": "cpu"}
    encode_kwargs: Dict[str, bool] = {'normalize_embeddings': False}


class ChromaSettings(BaseSettings):
    collection_name: Literal["tyuiu-documents"] = "tyuiu-documents"


class GigaChatSettings(BaseSettings):
    client_id: str = os.getenv("CLIENT_ID")
    client_secret: str = os.getenv("CLIENT_SECRET")
    auth_key: str = os.getenv("AUTH_KEY")
    scope: str = os.getenv("GIGACHAT_API_PERS")
    url: str = os.getenv("AUTH_URL")

    prompt: Path = BASE_DIR / "static" / "prompt" / "chat.txt"


class APISettings(BaseSettings):
    name: str = "RAG GigaChat API"
    prefix: str = "/api/v1"


class Settings(BaseSettings):
    api_v1: APISettings = APISettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    chroma: ChromaSettings = ChromaSettings()
    giga_chat: GigaChatSettings = GigaChatSettings()


settings = Settings()
