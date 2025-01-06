from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models.gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate

from src.rag.abstract.llm import AbstractModel
from src.rag.generator.auth import GigaChatAuth
from src.misc.file import load_txt

from langchain_core.runnables.base import Runnable


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


prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
Используй при этом только информацию из контекста. Если в контексте нет \
информации для ответа, сообщи об этом пользователю.
Контекст: {context}
Вопрос: {input}
Ответ:'''
)
document_chain = create_stuff_documents_chain(
    llm=GigaChatModel(GigaChatAuth().get_auth_key())._llm,
    prompt=prompt
)
from src.rag.vector_store.chroma import ChromaVectorStore
embedding_retriever = ChromaVectorStore().get_embeddings_retriever()
retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
q1 = "Как поступить в ВШЦТ"
resp1 = retrieval_chain.invoke(
    {'input': q1}
)
print(resp1["answer"])
