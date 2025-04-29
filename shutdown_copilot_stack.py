import subprocess
import time
import psutil
import socket

# -------- Process Match Keywords --------
OLLAMA_MATCH = "ollama"
FASTAPI_MATCH = "uvicorn fastapi_sql_copilot:app"
WSL_MATCH = "wsl"

# -------- Shutdown Helpers --------

def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            print(f"🔪 Killing child process (PID: {child.pid})")
            child.kill()
        print(f"🔪 Killing parent process (PID: {parent.pid})")
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

def kill_matching_processes(match_strings):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
            for match_string in match_strings:
                if match_string in cmdline:
                    print(f"🛑 Terminating process tree for {match_string} (PID: {proc.pid})")
                    kill_process_tree(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def kill_ollama_server():
    print("🛑 Stopping Ollama server...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and "ollama" in proc.info['name'].lower():
                print(f"🔪 Killing Ollama server (PID: {proc.pid})")
                kill_process_tree(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def stop_postgres_in_wsl():
    print("🔻 Stopping PostgreSQL inside WSL...")
    result = subprocess.run(["wsl", "-e", "bash", "-c", "sudo service postgresql stop"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ PostgreSQL stopped.")
    else:
        print("⚠️ Error stopping PostgreSQL:", result.stderr.strip())

def shutdown_wsl():
    print("🛑 Shutting down WSL2 completely...")
    result = subprocess.run(["wsl", "--shutdown"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ WSL shut down.")
    else:
        print("⚠️ WSL shutdown may have failed:", result.stderr.strip())

def kill_all_powershell_windows():
    print("🧹 Killing all open PowerShell windows (final cleanup)...")
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] and "powershell" in proc.info['name'].lower():
                print(f"🪟 Killing PowerShell window (PID: {proc.pid})")
                kill_process_tree(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def is_port_in_use(port=11434):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0

# --------- Main Shutdown Flow ---------

def main():
    print("\n🛑 Starting Copilot Stack Shutdown...\n")

    # 1. Stop FastAPI & Ollama processes
    kill_matching_processes([FASTAPI_MATCH])
    time.sleep(2)

    if is_port_in_use(11434):
        kill_ollama_server()
    else:
        print("✅ Ollama already stopped.")

    # 2. Stop PostgreSQL & shutdown WSL
   # stop_postgres_in_wsl()
    time.sleep(1)

  #  shutdown_wsl()
    time.sleep(1)

    # 3. Kill all remaining PowerShell windows
  #  kill_all_powershell_windows()

    print("\n✅ Copilot Stack fully shut down. All zombie PowerShells cleaned.\n")

if __name__ == "__main__":
    main()
