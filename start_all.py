import subprocess
import sys
import time
import os
import webbrowser
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_backend():
    print("\n[1/3] Starting FastAPI backend...")
    # Using sys.executable to ensure we use the same python environment
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="backend"
    )

def main():
    print("=== Health AI Agent Startup ===")
    
    # Step 1: Install requirements
    print("\n[0/3] Checking and installing dependencies (this may take a minute)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "transformers", "torch", "pandas", "scikit-learn", "joblib"])
    except Exception as e:
        print(f"Warning: Dependency installation might have failed: {e}")

    # Step 2: Start Backend
    if is_port_in_use(8000):
        print("Port 8000 is already in use. Assuming backend is already running.")
        backend_process = None
    else:
        backend_process = run_backend()
        print("Waiting for backend to initialize (loading BioBERT model)...")
        # Wait until port 8000 is active
        for _ in range(30):
            if is_port_in_use(8000):
                print("Backend is now active!")
                break
            time.sleep(2)
        else:
            print("Backend is taking a long time to start. Opening frontend anyway...")

    # Step 3: Open Frontend
    print("\n[2/3] Opening dashboard...")
    frontend_path = os.path.abspath("frontend/index.html")
    url = f"file:///{frontend_path.replace(os.sep, '/')}"
    webbrowser.open(url)
    
    print("\n[3/3] System Ready!")
    print(f"Frontend: {url}")
    print("Backend API: http://localhost:8000")
    print("\nKEEP THIS TERMINAL OPEN while using the dashboard.")
    print("Press Ctrl+C to stop the server.")

    try:
        if backend_process:
            backend_process.wait()
        else:
            # If backend was already running, just keep this script alive
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if backend_process:
            backend_process.terminate()

if __name__ == "__main__":
    main()
