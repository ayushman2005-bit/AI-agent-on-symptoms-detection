import subprocess
import sys
import time
import os
import webbrowser

def run_backend():
    print("Starting FastAPI backend...")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"], cwd="backend")

def main():
    # Install requirements if needed
    print("Checking dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])

    backend_process = None
    try:
        backend_process = run_backend()
        
        # Wait for backend to start
        print("Waiting for backend to initialize...")
        time.sleep(5)
        
        # Open the frontend
        frontend_path = os.path.abspath("frontend/index.html")
        print(f"Opening dashboard: {frontend_path}")
        webbrowser.open(f"file://{frontend_path}")
        
        print("\nHealth AI Agent is running!")
        print("Press Ctrl+C to stop.")
        
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping Health AI Agent...")
    finally:
        if backend_process:
            backend_process.terminate()

if __name__ == "__main__":
    main()
