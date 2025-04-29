# test_postgres_wsl_outside.py

import subprocess
import psycopg2
import socket

# Database credentials (Windows trying to connect to WSL PostgreSQL)
DB_USER = "admin"
DB_PASSWORD = "admin123"
DB_NAME = "service_catalog"
DB_PORT = 5432

def get_wsl_ip():
    print("🔍 Fetching WSL2 IP Address...")
    try:
        result = subprocess.run(["wsl", "-e", "bash", "-c", "hostname -I | awk '{print $1}'"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            ip = result.stdout.strip()
            print(f"✅ WSL IP detected: {ip}")
            return ip
        else:
            print(f"❌ Unable to fetch WSL IP: {result.stderr.strip()}")
            return None
    except Exception as e:
        print(f"❌ Error fetching WSL IP: {e}")
        return None

def test_tcp_connection(ip, port):
    print(f"🔍 Testing TCP connection to {ip}:{port}...")
    try:
        with socket.create_connection((ip, port), timeout=5):
            print(f"✅ TCP Connection to {ip}:{port} successful.")
            return True
    except Exception as e:
        print(f"❌ TCP Connection failed: {e}")
        return False

def test_postgres_connection(ip):
    print(f"🔍 Testing PostgreSQL DB connection to {ip}:{DB_PORT}...")
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=ip,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        if result and result[0] == 1:
            print("✅ PostgreSQL Query Success (SELECT 1).")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ PostgreSQL Connection failed: {e}")

def main():
    print("\n🚀 Testing PostgreSQL Connection from Windows to WSL...\n")

    wsl_ip = get_wsl_ip()
    if not wsl_ip:
        print("❌ Cannot continue without WSL IP.")
        return

    # Step 1: Test if TCP port 5432 is reachable
    if not test_tcp_connection(wsl_ip, DB_PORT):
        print("\n❌ PostgreSQL is not reachable over TCP.")
        print("👉 Check if PostgreSQL is running and listening on WSL IP.")
        return

    # Step 2: Test if database authentication works
    test_postgres_connection(wsl_ip)

    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
