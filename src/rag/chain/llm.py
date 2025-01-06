from langchain_core.runnables.base import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.gigachat import GigaChat
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.rag.abstract import AbstractModel


class GigaChatModel(AbstractModel):
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

    def create_documents_chain(self, template: str) -> Runnable:
        prompt = ChatPromptTemplate.from_template(template)
        document_chain = create_stuff_documents_chain(
            llm=self._llm,
            prompt=prompt
        )
        return document_chain
