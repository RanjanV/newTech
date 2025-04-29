# view_faiss_vectorstore.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# -------- Configuration --------
FAISS_INDEX_PATH = r"C:\LLM\faiss_index"  # Directory where your faiss_index/ is saved
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"❌ FAISS index directory '{FAISS_INDEX_PATH}' not found!")
        return

    print("🚀 Loading FAISS Vectorstore...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # 🔥 Required now
    )

    print("✅ Loaded FAISS index successfully.")
    print("📚 Fetching stored documents...")

    docs = vectorstore.docstore._dict

    if not docs:
        print("⚠️ No documents found inside the vectorstore!")
        return

    print(f"📄 Total documents stored: {len(docs)}\n")
    print("=" * 80)

    for idx, (doc_id, document) in enumerate(docs.items(), start=1):
        print(f"🔹 Document {idx}")
        print(f"ID: {doc_id}")
        print(f"Content:\n{document.page_content}")
        print("=" * 80)

        # Optional: Limit to first 10 documents
        #if idx >= 10:
        #    print("🔔 Showing only first 10 documents. (Limit reached)")
        #    break

if __name__ == "__main__":
    main()
