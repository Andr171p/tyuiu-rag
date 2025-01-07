import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal, Dict
from pydantic_settings import BaseSettings


BASE_DIR: Path = Path(__file__).resolve().parent.parent

ENV_PATH: Path = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


class EmbeddingsSettings(BaseSettings):
    model_name: str = os.getenv("EMBEDDINGS_MODEL_NAME")
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

    class PromptSettings(BaseSettings):
        path: Path = BASE_DIR / "static" / "prompt" / "chat.txt"

    prompt: PromptSettings = PromptSettings()


class Settings(BaseSettings):
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    chroma: ChromaSettings = ChromaSettings()
    giga_chat: GigaChatSettings = GigaChatSettings()


settings = Settings()
