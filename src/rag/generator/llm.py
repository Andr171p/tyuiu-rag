from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.gigachat import GigaChat
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.rag.abstract import AbstractAuth, AbstractLLM

if TYPE_CHECKING:
    from langchain_core.runnables.base import Runnable


class GigaChatLLM(AbstractLLM):
    def __init__(
            self,
            auth: AbstractAuth,
            model_name: str = "GigaChat:latest"
    ) -> None:
        self._llm = GigaChat(
            credentials=auth.get_auth_key(),
            model=model_name,
            verify_ssl_certs=False,
            profanity_check=False
        )

    def create_llm_chain(self, template: str) -> "Runnable":
        prompt = ChatPromptTemplate.from_template(template)
        document_chain = create_stuff_documents_chain(
            llm=self._llm,
            prompt=prompt
        )
        return document_chain


# prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
# Используй при этом только информацию из контекста. Если в контексте нет \
# информации для ответа, сообщи об этом пользователю.
# Контекст: {context}
# Вопрос: {input}
# Ответ:'''
# )
'''from src.rag.embeddings.model import EmbeddingsModel
document_chain = create_stuff_documents_chain(
    llm=GigaChatLLM(GigaChatAuth())._llm,
    prompt=prompt
)
from src.rag.retriever.chroma import ChromaRetriever
from src.rag.embeddings.model import EmbeddingsModel
embedding_retriever = ChromaRetriever(EmbeddingsModel().embeddings_function()).get_embeddings_retriever()
retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
q1 = "Как поступить в ВШЦТ"
resp1 = retrieval_chain.invoke(
    {'input': q1}
)
print(resp1["answer"])'''
