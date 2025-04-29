# build_rag_vectorstore.py

import os
import psycopg2
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import pandas as pd
import re

# Load environment variables
load_dotenv()

# PostgreSQL connection config
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin123")
DB_HOST = os.getenv("DB_HOST", "172.24.246.204")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "service_catalog")

# Local docs folder
DOCS_DIR = r"C:\LLM\refDocs"

# Utility: Extract service name from file name
def extract_service_name(filename):
    match = re.match(r"([a-zA-Z0-9\-]+-service)", filename)
    if match:
        return match.group(1)
    return None

# Step 1: Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# Step 2: Load Services
cur.execute("""SELECT id, name, description, contact, environment, diagram_url FROM services""")
service_rows = cur.fetchall()

service_documents = []
for row in service_rows:
    service_text = f"Service: {row[1]}\nDescription: {row[2]}\nContact: {row[3]}\nEnvironment: {row[4]}\nDiagram URL: {row[5]}"
    metadata = {
        "source": "services",
        "service_id": row[0],
        "service_name": row[1],
        "contact": row[3],
        "environment": row[4],
        "diagram_url": row[5]
    }
    service_documents.append(Document(page_content=service_text, metadata=metadata))

print(f"✅ Loaded {len(service_documents)} services.")

# Step 3: Load Documents from Files (Excel & Word)
file_documents = []

for filename in os.listdir(DOCS_DIR):
    filepath = os.path.join(DOCS_DIR, filename)
    service_name = extract_service_name(filename)

    try:
        if filename.endswith(".txt"):
            loader = TextLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["related_service"] = service_name
            file_documents.extend(docs)

        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["related_service"] = service_name
            file_documents.extend(docs)

        elif filename.endswith(".xlsx"):
            xls = pd.ExcelFile(filepath)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                content = df.to_string(index=False)
                metadata = {
                    "source": filename,
                    "sheet": sheet_name,
                    "related_service": service_name
                }
                file_documents.append(Document(
                    page_content=f"[{sheet_name}] for {service_name}\n{content}",
                    metadata=metadata
                ))
        else:
            continue

    except Exception as e:
        print(f"⚠️ Failed to load {filename}: {e}")

print(f"✅ Loaded {len(file_documents)} documents from files.")

# Step 4: Load APIs
cur.execute("""SELECT id, service_id, path, method, description FROM apis""")
api_rows = cur.fetchall()

api_documents = []
for row in api_rows:
    api_text = f"API Path: {row[2]}\nMethod: {row[3]}\nDescription: {row[4]}"
    metadata = {
        "source": "apis",
        "api_id": row[0],
        "service_id": row[1],
        "api_path": row[2],
        "api_method": row[3]
    }
    api_documents.append(Document(page_content=api_text, metadata=metadata))

print(f"✅ Loaded {len(api_documents)} APIs.")

# Step 5: Load Operations
cur.execute("""SELECT id, api_id, name, description FROM operations""")
operation_rows = cur.fetchall()

operation_documents = []
for row in operation_rows:
    operation_text = f"Operation Name: {row[2]}\nOperation Description: {row[3]}"
    metadata = {
        "source": "operations",
        "operation_id": row[0],
        "api_id": row[1],
        "operation_name": row[2]
    }
    operation_documents.append(Document(page_content=operation_text, metadata=metadata))

print(f"✅ Loaded {len(operation_documents)} operations.")

# Step 6: Close DB connection
cur.close()
conn.close()

# Step 7: Merge All Documents
all_documents = service_documents + api_documents + operation_documents + file_documents
print(f"🔹 Total {len(all_documents)} documents before chunking.")

# Step 8: Split into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
split_documents = text_splitter.split_documents(all_documents)
print(f"🔹 Total {len(split_documents)} chunks prepared for embedding.")

# Step 9: Generate Embeddings and Create FAISS Index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_documents, embedding_model)

# Step 10: Save FAISS Index
faiss_index_path = r"C:\LLM\faiss_index"
if os.path.exists(faiss_index_path):
    shutil.rmtree(faiss_index_path)

vectorstore.save_local(faiss_index_path)
print(f"✅ FAISS index saved with {len(split_documents)} chunks.")
