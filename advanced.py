from typing import Iterable, Any

from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever


with open(r"C:\Users\andre\TyuiuRAG\static\texts\ТИУ для абитуриентов.txt", encoding="utf-8") as file:
    big_doc = file.read()

length = len(big_doc)
print(f"Big document length: {length}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=500,
    length_function=len
)

texts = text_splitter.create_documents([big_doc])

from new_rag.retrievers.elastic_search import ElasticSearchRetrieverFactory
esr = ElasticSearchRetrieverFactory()
r = esr.create_retriever()
# r.add_texts([t.page_content for t in texts])
docs = r.invoke("Расскажи о Поступлении")
print(docs[0])

'''model_name = "d0rj/e5-base-en-ru"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}


class MyEmbeddingsModel(HuggingFaceEmbeddings):
    model_config = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "encode_kwargs": encode_kwargs
    }


embeddings = MyEmbeddingsModel()


db = Chroma.from_documents(texts, embeddings)
# docs = db.similarity_search("Расскажи про дополнительные баллы")
# print(docs[0])


chroma_retriever = db.as_retriever(search_kwargs={"k": 5})
bm25 = BM25Retriever.from_documents(texts)
retriever = EnsembleRetriever(
    retrievers=[bm25, chroma_retriever], weights=[0.4, 0.6]
)
docs = retriever.get_relevant_documents("Какие сть курсы для абитуриентов")
print(docs[0])
print(docs[1])
print(docs[2])'''