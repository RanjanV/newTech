
import os
os.environ["OLLAMA_BASE_URL"] = "https://acer-knock-workforce-blocks.trycloudflare.com"
from langchain_community.llms import Ollama
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv


# Load DB credentials from .env
load_dotenv()

DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin123")
DB_HOST = os.getenv("DB_HOST", "172.24.246.204")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "service_catalog")

# 1. Connect to PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
db = SQLDatabase(engine)

# 2. Initialize Ollama LLM (llama3.2)
llm = Ollama(model="llama3", base_url="https://acer-knock-workforce-blocks.trycloudflare.com/?token=e58663d5a3a69042e8be0065e8ec66225ef48fe02f34f5d4b2065e86869a5ea2",temperature=0.2)
# 3. Create the SQLDatabaseChain (LangChain SQL agent)
chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 4. Ask a natural language question
question = "List all services in production"
response = chain.run(question)

print("\n✅ Result:\n", response)
