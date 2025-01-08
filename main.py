from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# from langchain.vectorstores.faiss import FAISS

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

loader = UnstructuredWordDocumentLoader(r"C:\Users\andre\TyuiuRAG\static\docs\ТИУ основная информация.docx")
split_docx = loader.load_and_split(text_splitter)

# print(split_docx[1])

model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs,
                                  encode_kwargs=encode_kwargs)

# retriever = FAISS.from_documents(split_docx, embedding=embedding)
persist_directory = r"C:\Users\andre\TyuiuRAG\chroma"
db = Chroma.from_documents(
    split_docx,
    embedding,
    persist_directory=persist_directory
    # client_settings=Settings(anonymized_telemetry=False),
)
db.persist()

question = "Направления строительного института"
docs = db.similarity_search(question, k=4)
print(docs[0])


'''bm25_retriever = BM25Retriever.from_documents(
    ...
)'''
