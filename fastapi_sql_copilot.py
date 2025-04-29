import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="db_config.env")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
os.environ["OLLAMA_BASE_URL"] = f"{OLLAMA_HOST}"
# fastapi_sql_copilot.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.sql_database import SQLDatabase
import requests
import asyncio
import re
import logging
import time
import datetime

from typing import Any, Dict, List, Optional
from langchain_core.outputs import LLMResult, Generation
from sqlalchemy import text
from fastapi.responses import HTMLResponse

# -------------- Load Environment --------------

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

OLLAMA_COOKIE = os.getenv("OLLAMA_COOKIE")  # <-- Cookie from env

# -------------- Custom LLM with Cookie --------------
class CustomOllamaWithCookie(Ollama):
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> LLMResult:
        prompt = prompts[0]
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Cookie": f"C.19744611_auth_token={OLLAMA_COOKIE}"
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise Exception(f"Ollama call failed with status code {response.status_code}. Details: {response.text}")
        res_json = response.json()
        generations = [Generation(text=res_json["response"])]
        return LLMResult(generations=[generations])

# -------------- Setup Logging --------------
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler(r"C:\LLM\rag_conversations.log", mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
logger = logging.getLogger(__name__)

# -------------- Conversation Logging Setup --------------
conversation_logging_enabled = True
conversation_log_file = r"C:\LLM\rag_conversations.log"

def log_semantic_conversation(user_query, retrieved_context, final_prompt, llm_response):
    if not conversation_logging_enabled:
        return
    with open(conversation_log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("[Semantic Mode]\n")
        f.write(f"User Query: {user_query}\n\n")
        f.write("Retrieved Context Chunks:\n")
        for idx, doc in enumerate(retrieved_context, 1):
            f.write(f"  Chunk {idx}: {doc.page_content}\n")
        f.write("\nFinal Prompt Sent to Ollama:\n")
        f.write(final_prompt + "\n\n")
        f.write("LLM Response:\n")
        f.write(llm_response + "\n")
        f.write("="*80 + "\n")

def log_sql_conversation(user_query, sql_thoughts, sql_final_answer):
    if not conversation_logging_enabled:
        return
    with open(conversation_log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("[SQL Mode]\n")
        f.write(f"User Query: {user_query}\n\n")
        f.write("SQL Thoughts and Actions:\n")
        f.write(sql_thoughts + "\n")
        f.write("\nFinal Answer:\n")
        f.write(sql_final_answer + "\n")
        f.write("="*80 + "\n")

# -------------- FastAPI App --------------
app = FastAPI(title="Service Catalog Copilot - Clean SQL + RAG + Router")

# -------------- Initialize Everything --------------
# 1. LLM
llm = CustomOllamaWithCookie(
    model=OLLAMA_MODEL,
    base_url=f"{OLLAMA_HOST}",
    temperature=0.2,
    timeout=120,
)

# 2. VectorStore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(r"C:\LLM\faiss_index", embedding_model, allow_dangerous_deserialization=True)

# 3. RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, verbose=True)

# 4. Database
db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(db_uri, include_tables=["services", "apis", "operations"])

# 5. SQL Schema Injection
schema_description = db.get_table_info()
system_message = SystemMessagePromptTemplate.from_template(
    f"""You are a SQL expert working on a Postgres database.
Only use the following tables and columns:
{schema_description}

When writing SQL queries, only use the tables listed above.
Never invent table names or columns. If unsure, ask for clarification."""
)
'''
sql_prompt = PromptTemplate.from_template(f"""
You are a PostgreSQL expert.

Use only the following database schema:
{schema_description}

Important Rules:
- Always write a SQL query that starts with SELECT or WITH.
- NEVER use CREATE, INSERT, UPDATE, DELETE, DROP, ALTER, etc.
- Assume tables and data already exist.
- Only select and filter based on available columns.
- When checking if a field is missing (e.g., contact, diagram_url), always handle three cases:
    1. Field IS NULL
    2. Field is an empty string ('')
    3. Field has invalid values like 'NaN' (case-insensitive).
- When checking description of a service (e.g., service has billing info), always consider description column for query.
- Use conditions like: `WHERE field IS NULL OR TRIM(field) = '' OR LOWER(field) = 'nan'`
- If uncertain, ask the user for clarification — do not invent tables or fields.

User question: {{input}}

Respond with a complete valid SQL query only.
SQL Query:
""")
'''

sql_prompt = PromptTemplate.from_template(f"""
You are a PostgreSQL expert.

Use only the following database schema:
{schema_description}

Important Special Notes:
- Table 'services' → 'name' is service name.
- Table 'apis' → 'path' is API path (e.g., /login).
- Table 'apis' → 'method' is HTTP method (GET/POST).
- Table 'operations' → 'name' is operation name.

Important Rules:
- Always SELECT using real table and column names.
- Never use fake placeholders like 'table' or 'field'.
- Never invent new columns.
- Always SELECT from the correct table based on schema.
- Always use real SQL syntax.
- Always prefer simple SELECT queries if possible.
- Only SELECT/WHERE allowed (no CREATE, UPDATE, DELETE, ALTER).
- If uncertain, ask for clarification instead of guessing.

User question: {{input}}

Respond with a complete valid SQL query only. No explanation, no markdown formatting.

SQL Query:
""")



sql_chain = LLMChain(llm=llm, prompt=sql_prompt, verbose=True)

# 6. Router Chain
# 6. Router Chain (UPDATED)
router_prompt = PromptTemplate.from_template("""
Classify the following user question as one of:
1. structured_query → SQL-style database query (e.g., find missing fields like contact, environment, diagram_url, NULL checks, empty fields, incomplete data)
2. semantic_query → Meaning-based understanding (e.g., explain what auth-service does, or describe services)
3. faq_query → Help, access, or documentation questions (e.g., how to request access)

Important Notes:
- Any query that mentions missing fields, empty values, NULLs, incomplete records, or checks on fields must be classified as structured_query.
- Even if it looks like a natural language question, missing fields checking is always a structured_query.

Question: {query}
Answer (structured_query, semantic_query, or faq_query):
""")

router_chain = LLMChain(llm=llm, prompt=router_prompt)


clarification_prompt = PromptTemplate.from_template("""
You are an expert in rephrasing database-related questions cleanly.

Given a user question, rephrase it clearly and simply so that it can be directly used to generate a SQL query.

Important Rules:
- Focus on making the question short, direct, and precise.
- If HTTP methods like GET/POST are mentioned, retain them accurately.
- Never ask user to provide more input.
- Never explain your steps.
- Simply return the clean rephrased version of the user question.
- Do not add any examples or markdown formatting.

User question: {input}

Clarified clean version:
""")


clarification_chain = LLMChain(llm=llm, prompt=clarification_prompt)






# -------------- Models --------------
class Question(BaseModel):
    query: str

# -------------- Utility Functions --------------
def simple_sql_detector(query: str) -> bool:
    sql_keywords = ["list", "show", "give me", "find", "get", "display", "fetch"]
    simple_conditions = ["empty", "missing", "null", "blank", "without contact", "no contact"]
    query_lower = query.lower()
    return any(k in query_lower and c in query_lower for k in sql_keywords for c in simple_conditions)

def extract_sql_from_text(text: str) -> str:
    match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    select_match = re.search(r"(SELECT\s.*?;)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    return text.strip()

def execute_sql_query(sql: str):
    try:
        with db._engine.connect() as connection:
            result = connection.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
            return data
    except Exception as e:
        logger.error(f"❌ Error executing SQL: {e}")
        return {"error": str(e)}

def is_safe_sql(sql: str) -> bool:
    sql_clean = sql.strip().lower()
    return sql_clean.startswith("select") or sql_clean.startswith("with")

async def ping_ollama_periodically():
    while True:
        try:
            headers = {
                "Content-Type": "application/json",
                "Cookie": f"C.19744611_auth_token={OLLAMA_COOKIE}"
            }
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "hello", "stream": False},
                headers=headers,
                timeout=60
            )
            if response.status_code == 200:
                logger.info("✅ Pinged Ollama to keep model warm.")
        except Exception as e:
            logger.warning(f"⚠️ Ping failed: {str(e)}")
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_tasks():
    asyncio.create_task(ping_ollama_periodically())

# -------------- Main Endpoint --------------
# (Your /copilot_ask endpoint remains exactly as you wrote, no change needed there)

# -------------- Logs Viewer API --------------
# (also remains exactly as you wrote)



# -------------- Main Endpoint --------------
@app.post("/copilot_ask")
async def copilot_ask(item: Question):
    start_time = time.time()
    query = item.query.strip()
    logger.info(f"📥 Incoming Query: {query}")

    try:
        # Detect
        if simple_sql_detector(query):
            logger.info("⚡ Fast SQL Detected.")
            sql_response = await asyncio.wait_for(
                run_in_threadpool(lambda: sql_chain.run(query)),
                timeout=60
            )
            duration = round(time.time() - start_time, 2)
            logger.info(f"✅ Fast SQL Completed in {duration} sec")
            return {"type": "fast_sql_direct", "query": query, "answer": sql_response}

        # Route
        route = await asyncio.wait_for(run_in_threadpool(lambda: router_chain.run(query)), timeout=60)
        route = route.strip().lower()
        logger.info(f"🛣️ Router decided: {route}")

        if "structured" in route:
            logger.info(f"🛠️ Structured Query Match detected. Clarifying user query first...")

            # ✨ Step 1: Clarify user input, but do not generate SQL yet
            clarified_query = await asyncio.wait_for(
                run_in_threadpool(lambda: clarification_chain.run(query)),
                timeout=30
            )
            logger.info(f"✅ Clarified Query: {clarified_query}")

            duration = round(time.time() - start_time, 2)
            logger.info(f"✅ Clarification Completed in {duration} sec")

            # ✨ Step 2: Return clarification to user to ask for confirmation
            return {
                "type": "clarification",
                "original_query": query,
                "clarified_query": clarified_query,
                "message": f"I understood your request as: '{clarified_query}'. Should I proceed? (yes/no)"
            }

        elif "semantic" in route:
            logger.info(f"🔎 Semantic Query Match detected. Executing RAG Retriever...")

            logger.info(f"🔎 Starting RAG Retriever Execution (run_in_threadpool)...")
            try:
                rag_response = await asyncio.wait_for(
                    run_in_threadpool(lambda: rag_chain.invoke({"query": query})),
                    timeout=30
                )
                logger.info(f"🔎 RAG Retriever Execution Completed.")
            except asyncio.TimeoutError:
                logger.error(f"⏱️ RAG Retriever processing timeout.")
                raise HTTPException(status_code=504, detail="RAG Retriever timed out.")

            user_question = item.query
            retrieved_docs = rag_response["source_documents"]

            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            final_prompt = f"""Use the following pieces of context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context_text}

        Question:
        {user_question}

        Helpful Answer:"""

            llm_generated_answer = rag_response["result"]

            log_semantic_conversation(
                user_query=user_question,
                retrieved_context=retrieved_docs,
                final_prompt=final_prompt,
                llm_response=llm_generated_answer
            )

            duration = round(time.time() - start_time, 2)
            logger.info(f"✅ Semantic Query (RAG) Completed in {duration} sec")
            return {
                "type": "semantic_query",
                "query": user_question,
                "answer": llm_generated_answer,
                "sources": [doc.page_content for doc in retrieved_docs]
            }

        elif "faq" in route:
            logger.info(f"📖 FAQ/Help Query detected. Generating FAQ-style response...")
            return {
                "type": "faq_query",
                "query": query,
                "answer": "This appears to be a help or documentation question. Please refer to internal Service Catalog guidelines or raise a support request if needed."
            }

        else:
            logger.warning(f"⚠️ Unrecognized route or no match for query: {query}")
            return {
                "type": "unhandled_query",
                "query": query,
                "answer": "I'm sorry, I could not find an answer for your request in Service Catalog. Please refine your question or contact support."
            }

    except asyncio.TimeoutError:
        duration = round(time.time() - start_time, 2)
        logger.error(f"⏱️ Timeout after {duration} sec")
        raise HTTPException(status_code=504, detail="Request timed out.")

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        logger.error(f"❌ Exception after {duration} sec: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------- Logs Viewer API --------------
@app.get("/logs/view", response_class=PlainTextResponse)
async def view_logs():
    try:
        if not os.path.exists(conversation_log_file):
            return "🛑 No logs found yet."
        with open(conversation_log_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Error reading log file: {str(e)}")
        return f"❌ Error reading logs: {str(e)}"
# -------------- Chat HTML --------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("chat.html", "r", encoding="utf-8") as f:
        return f.read()
class ConfirmRequest(BaseModel):
    confirm: str
    clarified_query: str

@app.post("/copilot_confirm")
async def copilot_confirm(confirm_data: ConfirmRequest):
    try:
        confirm = confirm_data.confirm.strip().lower()
        clarified_query = confirm_data.clarified_query.strip()

        if confirm not in ["yes", "no"]:
            return {
                "type": "confirmation_error",
                "message": "Please respond with 'yes' or 'no'."
            }

        if confirm == "no":
            return {
                "type": "clarification_failed",
                "message": "Okay, please reframe your query and try again."
            }

        # If user confirms "yes"
        logger.info(f"✅ User confirmed. Proceeding with clarified query: {clarified_query}")

        # Step 1: Generate SQL from clarified query
        sql_response_text = await asyncio.wait_for(
            run_in_threadpool(lambda: sql_chain.run(clarified_query)),
            timeout=60
        )
        logger.info(f"✅ SQL generation completed.")

        # Step 2: Extract SQL
        extracted_sql = extract_sql_from_text(sql_response_text)
        logger.info(f"✅ Extracted SQL: {extracted_sql}")

        # Step 3: Check SQL safety
        if not is_safe_sql(extracted_sql):
            logger.error(f"❌ Unsafe SQL detected. Blocking execution.")
            raise HTTPException(status_code=400, detail="Unsafe SQL detected. Only SELECT queries are allowed.")

        # Step 4: Execute SQL
        db_results = execute_sql_query(extracted_sql)

        if db_results is None:
            return {
                "type": "sql_error",
                "message": "❌ Error executing SQL. Please try refining your query."
            }

        # Step 5: Format final answer
        if db_results:
            # Dynamic field detection
            if isinstance(db_results[0], dict):
                if 'name' in db_results[0]:
                    extracted_field = 'name'
                elif 'path' in db_results[0]:
                    extracted_field = 'path'
                else:
                    extracted_field = None

                if extracted_field:
                    values_list = [row.get(extracted_field, '') for row in db_results]
                    answer_text = "Here are the " + extracted_field + "s: " + ", ".join(values_list)
                else:
                    answer_text = "Here are the matching records."
            else:
                answer_text = "Unexpected data format in DB results."
        else:
            answer_text = "No records found matching your criteria."

        return {
            "type": "structured_query_result",
            "clarified_query": clarified_query,
            "generated_sql": extracted_sql,
            "answer": answer_text,
            "raw_db_results": db_results
        }

    except Exception as e:
        logger.error(f"❌ Exception in confirmation step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
