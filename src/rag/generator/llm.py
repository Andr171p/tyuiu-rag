from langchain_community.chat_models.gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate

from src.rag.generator.credentials import GigaChatCredentials


class GigaChatModel:
    def __init__(
            self,
            auth_key: str,
            model_name: str = "GigaChat:latest"
    ) -> None:
        self._llm = GigaChat(
            credentials=auth_key,
            model=model_name,
            verify_ssl_certs=False,
            profanity_check=False
        )


print(GigaChatModel(auth_key=GigaChatCredentials().get_auth_key()))