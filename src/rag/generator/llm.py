from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models.gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate

from src.rag.generator.credentials import GigaChatCredentials
from src.misc.file import load_txt


class GigaChatModel:
    def __init__(
            self,
            auth_key: str,
            model_name: str = "GigaChat:latest"
    ) -> None:
        self.llm = GigaChat(
            credentials=auth_key,
            model=model_name,
            verify_ssl_certs=False,
            profanity_check=False
        )

    async def add_prompt(self) -> None:
        ...


prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
Используй при этом только информацию из контекста. Если в контексте нет \
информации для ответа, сообщи об этом пользователю.
Контекст: {context}
Вопрос: {input}
Ответ:'''
)
document_chain = create_stuff_documents_chain(
    llm=GigaChatModel(GigaChatCredentials().get_auth_key()).llm,
    prompt=prompt
)
from src.rag.vector_store.chroma import ChromaDB
embedding_retriever = ChromaDB().get_embedding_retriever()
retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
q1 = "Как поступить в ВШЦТ"
resp1 = retrieval_chain.invoke(
    {'input': q1}
)
print(resp1)
