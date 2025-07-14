# chat_db

💬 Natural Language Chat App for MySQL Database (with RAG)
This project enables anyone — technical or non-technical — to query a live MySQL database using plain English, thanks to OpenAI's GPT model and FAISS-based Retrieval-Augmented Generation (RAG).

It translates natural language questions into SQL, executes them, and returns results both as a table and a human-readable explanation.

🚀 Features
🔍 Natural Language to SQL with GPT-3.5

✅ SQL Execution on live MySQL databases

📊 Tabular Results and plain English explanations

⚡ RAG Integration (via FAISS) to reduce repetitive API calls

🧠 Auto schema loading from your database

💻 Streamlit UI — clean, accessible, and fast


🛠️ Tech Stack:
Tool	Purpose
Python	Core backend and logic
Streamlit	UI for interaction
MySQL	Primary database
SQLAlchemy	Database connection
OpenAI GPT-3.5	Natural language to SQL + explanation
FAISS	Vector DB for RAG to cache previous queries
dotenv	Secure config via .env
LangChain	RAG orchestration with FAISS


📁 Folder Structure:
chat_mysql/
│
├── main.py               # Streamlit App (RAG + SQL logic)
├── .env                  # Sensitive config (ignored via .gitignore)
├── requirements.txt      # All dependencies
├── faiss_index/          # FAISS vector store (auto-created)
├── .gitignore            # Ignore sensitive & heavy files
└── README.md             # Project overview


🔐 .env Configuration:
Create a .env file with the following:
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database_name
OPENAI_API_KEY=your_openai_api_key


⚙️ Setup Instructions:
1. Clone the repo

git clone https://github.com/yourusername/chat_mysql.git
cd chat_mysql

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run main.py

############################

💡 Why RAG?
Using Retrieval-Augmented Generation with FAISS:

Caches previous query/response pairs

Reduces OpenAI API costs

Improves response speed for repeated questions


📌 To-Do / Improvements:
Add authentication

Enable multilingual queries

Support joins across multiple databases

Export results to CSV/Excel

Cloud deployment (EC2 / Azure App Service)