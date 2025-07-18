# rag_utils.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from schema_utils import get_schema_info
from rag_vector_utils import get_or_create_vectorstore, retrieve_relevant_schema

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Use API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_rag_sql_answer(engine, db_name, question):
    schema = get_schema_info(engine, force_refresh=True)
    schema_str = "\n".join([f"{table}: {', '.join(columns)}" for table, columns in schema.items()])

    vectorstore = get_or_create_vectorstore(schema_str, db_name)
    relevant_schema = retrieve_relevant_schema(vectorstore, question)

    prompt = f"""
You are a senior SQL data analyst.

Your job is to translate a user's natural language question into a **valid and safe MySQL query**, using only the schema provided.

Rules:
- Use only SELECT queries.
- If the user asks for "how many", assume aggregation like COUNT().
- If the question mentions "each", "per", "group", "by department", or similar, use GROUP BY.
- Use table/column names only from the schema — don’t guess.
- If the question is ambiguous, assume the user wants the most relevant columns.

SCHEMA:
{relevant_schema}

QUESTION:
{question}

Write only the SQL query below:
""".strip()


    sql_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()

    if not sql_response.lower().startswith("select"):
        return None

    # Add explanation
    explain_prompt = f"Explain in simple terms what this SQL query does:\n\n{sql_response}"
    explanation = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": explain_prompt}]
    ).choices[0].message.content.strip()

    return {
        "sql": sql_response,
        "explanation": explanation
    }
