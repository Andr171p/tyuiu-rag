from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from src.config import BASE_DIR

model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs,
                                  encode_kwargs=encode_kwargs)

path: str = str(BASE_DIR / "chroma")

chroma_db = Chroma(
    persist_directory=path,
    embedding_function=embeddings_model
)

retriever = chroma_db.as_retriever(
    embeddings_model=embeddings_model
)
resp = retriever.invoke("Индивидуальные достижение (Дополнительные баллы)")
print(resp)