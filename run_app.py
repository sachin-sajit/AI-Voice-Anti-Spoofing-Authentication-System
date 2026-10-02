import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "voice_anti_spoofing"
APP_PATH = BASE_DIR / "app" / "app.py"

def main():
    print("====================================================")
    print(" Launching AI Voice Anti-Spoofing & Authentication ")
    print("====================================================")
    print(f"Streamlit App Path: {APP_PATH}\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.port=8501"]
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error launching Streamlit application: {e}")

if __name__ == "__main__":
    main()
