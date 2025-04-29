# launch_copilot_stack.py

import subprocess
import time
import os
import webbrowser
import psutil
import datetime

# --------- Configuration ---------
OLLAMA_DIR = r"C:\LLM"
SCRIPT_DIR = r"C:\LLM"
PYTHON_EXEC = "python"
FASTAPI_APP_FILE = "fastapi_sql_copilot.py"
FASTAPI_MODULE = "fastapi_sql_copilot:app"
BUILD_SCRIPT = "build_rag_vectorstore.py"
FASTAPI_URL = "http://localhost:8000/docs"

# --------- Utils ---------

def log_step(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def is_process_running(process_name):
    """
    Check if there is any running process that contains the given name.
    """
    log_step(f"🔍 Checking if process '{process_name}' is running...")
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline and isinstance(cmdline, list):
                command_line = " ".join(cmdline).lower()
                if process_name.lower() in command_line:
                    log_step(f"✅ Found running process: {command_line}")
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    log_step(f"⚡ No process '{process_name}' found running.")
    return False

# --------- Launcher Functions ---------

def start_wsl2_and_postgres():
    start = time.time()
    log_step("🚀 Checking PostgreSQL in WSL...")
    status = False
    if not is_process_running("postgres"):
        log_step("✅ PostgreSQL not running. Starting inside WSL...")
        powershell_command = '''
        Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoExit', '-Command', "wsl sudo service postgresql start; echo '✅ PostgreSQL started inside WSL'; bash"
        '''
        subprocess.Popen(["powershell.exe", "-Command", powershell_command])
        status = True
    else:
        log_step("⚡ PostgreSQL already running, skipping start.")
        status = True
    end = time.time()
    return ("PostgreSQL", status, round(end - start, 2))

def start_ollama_server():
    start = time.time()
    log_step("🚀 Checking Ollama server...")
    status = False
    if not is_process_running("ollama"):
        log_step("✅ Ollama not running. Starting server...")
        powershell_command = f'''
        Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoExit', '-Command', "cd '{OLLAMA_DIR}'; ollama serve"
        '''
        subprocess.Popen(["powershell.exe", "-Command", powershell_command])
        status = True
    else:
        log_step("⚡ Ollama server already running, skipping start.")
        status = True
    end = time.time()
    return ("Ollama", status, round(end - start, 2))

def run_build_vectorstore_script():
    start = time.time()
    log_step(f"🐍 Running {BUILD_SCRIPT}...")
    try:
        subprocess.run([PYTHON_EXEC, os.path.join(SCRIPT_DIR, BUILD_SCRIPT)], shell=True, check=True)
        log_step("✅ FAISS vectorstore built successfully.")
        status = True
    except subprocess.CalledProcessError as e:
        log_step(f"❌ Error running {BUILD_SCRIPT}: {e}")
        status = False
    end = time.time()
    return ("FAISS Build", status, round(end - start, 2))

def start_fastapi_server():
    start = time.time()
    log_step("🚀 Checking FastAPI server (uvicorn)...")
    status = False
    if not is_process_running("uvicorn"):
        log_step("✅ FastAPI server not running. Starting Uvicorn...")
        powershell_command = f'''
        Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoExit', '-Command', "cd '{SCRIPT_DIR}'; uvicorn {FASTAPI_MODULE} --reload"
        '''
        subprocess.Popen(["powershell.exe", "-Command", powershell_command])
        status = True
    else:
        log_step("⚡ FastAPI server already running, skipping start.")
        status = True
    end = time.time()
    return ("FastAPI", status, round(end - start, 2))

def open_fastapi_browser():
    log_step(f"🌐 Opening FastAPI Swagger UI: {FASTAPI_URL}")
    webbrowser.open(FASTAPI_URL)

def get_wsl_ip():
    log_step("🌐 Getting WSL IP address...")
    try:
        result = subprocess.run(["wsl", "-e", "bash", "-c", "hostname -I | awk '{print $1}'"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            ip = result.stdout.strip()
            log_step(f"📍 WSL IP Address: {ip}")
            return ip
        else:
            log_step(f"⚠️ Unable to fetch WSL IP: {result.stderr.strip()}")
            return None
    except Exception as e:
        log_step(f"❌ Error getting WSL IP: {e}")
        return None

# --------- Main Launcher ---------

def main():
    print("\n🔥 Launching Full Copilot Stack with Smart Checks...\n")

    results = []

    results.append(start_wsl2_and_postgres())
    time.sleep(8)

   # results.append(start_ollama_server())
   # time.sleep(5)

    get_wsl_ip()
    time.sleep(2)

    results.append(run_build_vectorstore_script())
    time.sleep(2)

    results.append(start_fastapi_server())
    time.sleep(3)

    open_fastapi_browser()

    print("\n✅ Copilot Stack fully launched and visible in multiple windows!\n")

    # Summary
    print("📋 Launch Summary:")
    print("-" * 40)
    print(f"{'Component':<15} {'Status':<10} {'Time Taken (s)':<15}")
    print("-" * 40)
    for comp, status, duration in results:
        status_text = "✅ Success" if status else "❌ Failed"
        print(f"{comp:<15} {status_text:<10} {duration:<15}")
    print("-" * 40)

if __name__ == "__main__":
    main()
