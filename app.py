import sys
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.gradio_ui import demo

if __name__ == "__main__":
    # Start FastAPI backend in background
    subprocess.Popen(
        ["uvicorn", "src.config.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for backend to start
    time.sleep(3)
    
    # Launch Gradio UI
    demo.launch()