# main.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import openai
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import urllib.parse
from dotenv import load_dotenv
from schema_utils import get_schema_info
from rag_utils import get_rag_sql_answer

# ------------------ Load Environment Variables ------------------ #
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------ Session State Initialization ------------------ #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "selected_db" not in st.session_state:
    st.session_state.selected_db = None
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "visual_config" not in st.session_state:
    st.session_state.visual_config = {}

# ------------------ Helper Functions ------------------ #
def list_databases():
    try:
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/")
        with engine.connect() as conn:
            result = conn.execute(text("SHOW DATABASES;"))
            return [row[0] for row in result]
    except Exception as err:
        st.error(f"[!] Failed to load databases: {err}")
        return []

def execute_sql(database, query):
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{database}")
    with engine.connect() as conn:
        result = conn.execute(text(query))
        columns = result.keys()
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

# ------------------ Main App ------------------ #
def main():
    st.title("Chat with Your MySQL Database + RAG")

    dbs = list_databases()
    selected_db = st.sidebar.selectbox("Select Database:", dbs)

    if selected_db != st.session_state.selected_db:
        st.session_state.selected_db = selected_db
        st.session_state.last_df = None

    if selected_db not in st.session_state.chat_history:
        st.session_state.chat_history[selected_db] = []

    # Load schema freshly each time to avoid mismatches
    try:
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{selected_db}")
        schema = get_schema_info(engine, force_refresh=True)
    except Exception as e:
        st.error(f"[!] Failed to load schema: {e}")
        schema = {}

    # Debug view (dev only)
    with st.expander("Show Schema (Debug View)"):
        st.json(schema)

    user_input = st.text_input("Ask a question about your database:", key="user_input")

    if st.button("Ask") and user_input:
        with st.spinner("Generating SQL and fetching data using RAG..."):
            rag_result = get_rag_sql_answer(engine, selected_db, user_input, schema)

            if not rag_result or not rag_result["sql"].lower().startswith("select"):
                st.error("[!] Only SELECT queries are allowed or valid SQL could not be generated.")
                return

            try:
                results = execute_sql(selected_db, rag_result["sql"])
                df = pd.DataFrame(results)
                st.session_state.last_df = df
                st.session_state.visual_config[selected_db] = {}

                st.success("Query executed successfully! Results:")
                st.write("Generated SQL Query:")
                st.code(rag_result["sql"], language="sql")
                st.dataframe(df)
                st.caption(f"Returned {len(df)} rows")

                st.write("Explanation:")
                st.info(rag_result["explanation"])

                st.session_state.chat_history[selected_db].append({
                    "question": user_input,
                    "sql": rag_result["sql"],
                    "explanation": rag_result["explanation"]
                })

            except Exception as e:
                st.error(f"Error executing query:\n\n```sql\n{rag_result['sql']}\n```\n\nError: {e}")

    # Visualization
    if st.session_state.last_df is not None:
        df = st.session_state.last_df
        with st.expander("Visualize the result"):
            config = st.session_state.visual_config[selected_db]

            chart_type = st.selectbox("Choose chart type", ["Bar", "Line", "Pie"], key="chart_type")
            lib = st.radio("Library", ["Seaborn", "Plotly"], horizontal=True, key="lib")
            x_axis = st.selectbox("X-axis", df.columns, key="x_axis")
            y_axis = st.selectbox("Y-axis", df.select_dtypes(include='number').columns, key="y_axis")

            if chart_type and x_axis and y_axis:
                st.subheader(f"{chart_type} Chart ({lib})")
                if lib == "Seaborn":
                    fig, ax = plt.subplots()
                    if chart_type == "Bar":
                        sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif chart_type == "Line":
                        sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif chart_type == "Pie":
                        df_pie = df.groupby(x_axis)[y_axis].sum().reset_index()
                        ax.pie(df_pie[y_axis], labels=df_pie[x_axis], autopct="%1.1f%%")
                        ax.axis("equal")
                    st.pyplot(fig)
                else:
                    if chart_type == "Bar":
                        fig = px.bar(df, x=x_axis, y=y_axis)
                    elif chart_type == "Line":
                        fig = px.line(df, x=x_axis, y=y_axis)
                    elif chart_type == "Pie":
                        df_pie = df.groupby(x_axis)[y_axis].sum().reset_index()
                        fig = px.pie(df_pie, values=y_axis, names=x_axis)
                    st.plotly_chart(fig, use_container_width=True)

    # Chat history
    with st.expander("Chat History"):
        chat_items = st.session_state.chat_history[selected_db]
        for item in chat_items:
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**SQL:**\n```sql\n{item['sql']}\n```")
            st.markdown(f"**Explanation:** {item['explanation']}")
            st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear History"):
                st.session_state.chat_history[selected_db] = []
                st.experimental_rerun()
        with col2:
            st.download_button("Download History", 
                               data=pd.DataFrame(chat_items).to_csv(index=False), 
                               file_name=f"{selected_db}_chat_history.csv", 
                               mime="text/csv")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey;'>Powered by OpenAI + LangChain · Built with ❤️ using Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()