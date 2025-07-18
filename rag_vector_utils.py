# rag_vector_utils.py
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

VECTOR_DIR = "vectorstores"

def get_or_create_vectorstore(schema_str, db_name):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vector_path = os.path.join(VECTOR_DIR, f"{db_name}_schema.index")

    if os.path.exists(vector_path):
        return FAISS.load_local(vector_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # Create new vectorstore
    docs = [Document(page_content=schema_str)]
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    vs.save_local(vector_path)
    return vs

def retrieve_relevant_schema(vectorstore, question, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    return "\n".join([doc.page_content for doc in docs])
