import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Database credentials
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create SQLAlchemy engine
def create_engine_with_db(db_name):
    return create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db_name}")

def list_databases():
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/")
    with engine.connect() as conn:
        result = conn.execute(text("SHOW DATABASES;"))
        return [row[0] for row in result]

def get_schema(db_name):
    engine = create_engine_with_db(db_name)
    query = """
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = :db_name;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"db_name": db_name})
        rows = result.fetchall()
    schema_desc = {}
    for table, col, dtype in rows:
        schema_desc.setdefault(table, []).append(f"{col} ({dtype})")
    schema_str = ""
    for table, cols in schema_desc.items():
        schema_str += f"Table `{table}` with columns: {', '.join(cols)}.\n"
    return schema_str

def get_or_create_vectorstore(schema, index_path):
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(schema)]
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(index_path)
        return vectorstore

def get_similar_context(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    return "\n".join([doc.page_content for doc in docs])

def generate_sql_and_response(prompt, context):
    messages = [
        {"role": "system", "content": f"You are an AI assistant that writes correct SQL queries for a MySQL database. Use the schema context:\n{context}"},
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

def run_sql_query(sql, db_name):
    try:
        engine = create_engine_with_db(db_name)
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
        {"role": "user", "content": f"SQL query: {sql}\nResult: {query_result}\nExplain the result."}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="Chat with MySQL", page_icon="💬")
    st.title("💬 Chat with Your MySQL Database")

    # Session states
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "selected_db" not in st.session_state:
        st.session_state.selected_db = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    st.sidebar.header("Database Connection")
    dbs = list_databases()
    selected_db = st.sidebar.selectbox("Select Database:", dbs)

    if st.sidebar.button("🔌 Connect"):
        try:
            schema = get_schema(selected_db)
            st.session_state.selected_db = selected_db
            st.session_state.connected = True
            index_path = f"faiss_index_{selected_db}"
            st.session_state.vectorstore = get_or_create_vectorstore(schema, index_path)
            st.success(f"✅ Connected to `{selected_db}`")
        except Exception as e:
            st.session_state.connected = False
            st.error(f"❌ Failed to connect: {e}")

    if not st.session_state.connected:
        st.warning("⚠️ Please connect to a database first.")
        return

    user_question = st.text_input("Ask a question about your database:")
    if user_question:
        st.session_state.chat_history.append(user_question)

        with st.spinner("Generating SQL..."):
            context = get_similar_context(user_question, st.session_state.vectorstore)
            generated_sql = generate_sql_and_response(user_question, context)

        st.markdown(f"🧠 **Question:** {user_question}")
        st.code(generated_sql, language="sql")  # ✅ Show the generated SQL query

        if generated_sql.lower().strip().startswith(("select", "show", "with", "insert", "update", "delete")):
            with st.spinner("Executing SQL query..."):
                results, error = run_sql_query(generated_sql, st.session_state.selected_db)

            if error:
                st.error(f"❌ SQL Error: {error}")
            else:
                if results:
                    df = pd.DataFrame(results)
                    st.success("✅ Query executed successfully! Results:")
                    st.dataframe(df)
                else:
                    st.warning("⚠️ Query executed successfully but returned no results.")

                with st.spinner("Generating explanation..."):
                    explanation = generate_natural_language_response(generated_sql, results)
                st.markdown(f"🗣️ **Explanation:**\n{explanation}")
        else:
            # Not a valid SQL query
            st.markdown(f"📝 **Response:**\n{generated_sql}")

    # Chat history
    if st.session_state.chat_history:
        st.markdown("🕓 **Chat History**")
        for q in reversed(st.session_state.chat_history):
            st.markdown(f"- {q}")

if __name__ == "__main__":
    main()
