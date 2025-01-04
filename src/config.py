import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal, List, Dict
from pydantic_settings import BaseSettings


BASE_DIR: Path = Path(__file__).resolve().parent.parent

ENV_PATH: Path = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


class EmbeddingsSettings(BaseSettings):
    model_name: str = os.getenv("MODEL_NAME")
    model_kwargs: Dict[str, str] = {"device": "cpu"}
    encode_kwargs: Dict[str, bool] = {'normalize_embeddings': False}


class ChromaSettings(BaseSettings):
    collection_name: Literal["tyuiu-documents"] = "tyuiu-documents"


class Settings(BaseSettings):
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    chroma: ChromaSettings = ChromaSettings()


settings = Settings()

