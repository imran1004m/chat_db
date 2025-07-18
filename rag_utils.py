from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_rag_sql_answer(engine, database, question, schema):
    context_lines = []
    for table, columns in schema.items():
        context_lines.append(f"Table `{table}`: columns = {', '.join(columns)}")
    context = "\n".join(context_lines)

    prompt = f"""
You are a MySQL expert assistant. Given the database schema and a natural language question, generate only a valid SELECT SQL query.

Instructions:
- Do NOT use INSERT, UPDATE, DELETE, DROP, or any modifying queries.
- Only generate a SELECT query using the correct table and column names from the schema.
- Do NOT include any markdown, explanation, or comments in your answer.
- You are querying the `{database}` MySQL database.

Schema:
{context}

Question:
{question}

Only return the SQL query (nothing else):
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": "You are a MySQL assistant that returns only valid SELECT queries." },
                { "role": "user", "content": prompt }
            ]
        )
        sql = response.choices[0].message.content.strip()

        # ✅ Ensure it's a SELECT query
        if not sql.lower().startswith("select"):
            return None  # causes UI to trigger warning

        # ✅ Explanation
        explanation_prompt = f"Explain the following SQL query in simple terms:\n\n{sql}"
        explanation_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": "You explain SQL queries clearly and simply." },
                { "role": "user", "content": explanation_prompt }
            ]
        )
        explanation = explanation_response.choices[0].message.content.strip()

        return {
            "sql": sql,
            "explanation": explanation
        }

    except Exception as e:
        print(f"[RAG ERROR] Failed to generate SQL: {e}")
        return None
