import requests

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # or whatever model you loaded

system_prompt = """
You are a data assistant. Convert the following natural language question into a SQL query for a PostgreSQL database 
with tables: services(id, name, description, contact, environment, diagram_url), 
apis(id, service_id, path, method, description), 
operations(id, api_id, name, description).
"""

question = "where is france located?"
prompt = f"{system_prompt}\nQuestion: {question}"

response = requests.post(
    f"{OLLAMA_URL}/api/generate",
    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
)

if response.status_code == 200:
    result = response.json().get("response", "")
    print("Generated SQL:\n", result.strip())
else:
    print("Error:", response.status_code, response.text)
