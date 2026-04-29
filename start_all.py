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
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="backend"
    )

def main():
    print("=== Health AI Agent Startup ===")

    # Step 0: Install requirements
    print("\n[0/3] Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "fastapi", "uvicorn", "transformers", "torch",
            "pandas", "scikit-learn", "joblib"
        ])
        print("Dependencies OK.")
    except Exception as e:
        print(f"Warning: Dependency install may have failed: {e}")

    # Step 1: Start backend
    if is_port_in_use(8000):
        print("Port 8000 already in use — assuming backend is already running.")
        backend_process = None
    else:
        backend_process = run_backend()
        print("Waiting for backend to start (BioBERT loading can take 1–5 minutes)...")
        # Wait up to 5 minutes (150 × 2s)
        for i in range(150):
            if is_port_in_use(8000):
                print(f"Backend is active! (took ~{(i+1)*2}s)")
                break
            if i % 15 == 14:
                print(f"  Still loading... ({(i+1)*2}s elapsed). BioBERT is large, please be patient.")
            time.sleep(2)
        else:
            print("Backend took over 5 minutes. Check for errors above. Opening frontend anyway...")

    # Step 2: Open frontend
    print("\n[2/3] Opening dashboard in browser...")
    frontend_path = os.path.abspath("frontend/index.html")
    url = f"file:///{frontend_path.replace(os.sep, '/')}"
    webbrowser.open(url)

    print("\n[3/3] System ready!")
    print(f"  Frontend : {url}")
    print(f"  Backend  : http://localhost:8000")
    print(f"  Health   : http://localhost:8000/health")
    print("\nKEEP THIS TERMINAL OPEN while using the dashboard.")
    print("Press Ctrl+C to stop.\n")

    try:
        if backend_process:
            backend_process.wait()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if backend_process:
            backend_process.terminate()
            print("Backend stopped.")

if __name__ == "__main__":
    main()
