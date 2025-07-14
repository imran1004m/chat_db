import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

load_dotenv()

# Load credentials
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Set up SQLAlchemy engine
DB_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URI)

INDEX_PATH = "faiss_index"

def get_schema():
    query = """
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = :db_name;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"db_name": DB_NAME})
        rows = result.fetchall()
    schema_desc = {}
    for table, col, dtype in rows:
        schema_desc.setdefault(table, []).append(f"{col} ({dtype})")
    schema_str = ""
    for table, cols in schema_desc.items():
        schema_str += f"Table `{table}` with columns: {', '.join(cols)}.\n"
    return schema_str

def get_or_create_vectorstore(schema: str):
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(schema)
        docs = [Document(page_content=t) for t in texts]
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(INDEX_PATH)
        return vectorstore

def generate_sql_and_response(prompt, schema):
    messages = [
        {"role": "system", "content": f"You are an AI assistant that writes correct SQL queries for a MySQL database. Here is the database schema:\n{schema}"},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    sql_match = re.search(r"```sql\n(.*?)```", content, re.DOTALL)
    return sql_match.group(1).strip() if sql_match else content.strip()

def run_sql_query(sql):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
        return data, None
    except Exception as e:
        return None, str(e)

def generate_natural_language_response(sql, query_result):
    messages = [
        {"role": "system", "content": "You are an assistant that explains SQL query results in simple natural language."},
        {"role": "user", "content": f"SQL query: {sql}\nResult: {query_result}\nExplain the result in a clear, simple way."}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def main():
    st.title("\U0001F4AC Chat with Your MySQL Database")
    schema = get_schema()
    st.sidebar.header("Database Schema")
    st.sidebar.text(schema)

    user_question = st.text_input("Ask a question about your database:")

    if user_question:
        with st.spinner("Setting up vector store (RAG)..."):
            vectorstore = get_or_create_vectorstore(schema)

        with st.spinner("Generating SQL query..."):
            retrieved_docs = vectorstore.similarity_search(user_question, k=2)
            retrieved_context = "\n".join(doc.page_content for doc in retrieved_docs)
            sql_query = generate_sql_and_response(user_question, retrieved_context)
        st.markdown(f"**Generated SQL Query:**\n```sql\n{sql_query}\n```")

        with st.spinner("Running query..."):
            results, error = run_sql_query(sql_query)

        if error:
            st.error(f"❌ SQL Error: {error}")
        else:
            if results:
                st.success("✅ Query executed successfully! Results:")
                df = pd.DataFrame(results)
                st.table(df)
            else:
                st.warning("⚠️ Query executed successfully but returned no results.")

            with st.spinner("Explaining results..."):
                explanation = generate_natural_language_response(sql_query, results)
            st.markdown(f"**Explanation:**\n{explanation}")

if __name__ == "__main__":
    main()
