from langchain.chat_models.gigachat import GigaChat

from src.rag.generator.credentials import GigaChatCredentials


class GigaChatModel:
    def __init__(self, credentials: str) -> None:
        self._llm = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            profanity_check=False
        )