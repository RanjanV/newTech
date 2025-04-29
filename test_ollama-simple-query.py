import requests

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # Change if you're using a different model

# Your question
question = "Where is France?"

# Call Ollama
response = requests.post(
    f"{OLLAMA_URL}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": question,
        "stream": False
    }
)

# Parse and print the answer
if response.status_code == 200:
    print("Answer from Ollama:\n")
    print(response.json().get("response", "").strip())
else:
    print("Error:", response.status_code, response.text)
