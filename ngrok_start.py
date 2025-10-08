# ngrok_start.py
from pyngrok import ngrok
import subprocess
import sys
import time

def start_with_ngrok():
    # Start the FastAPI server in background
    print("🚀 Starting FastAPI server...")
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "main:app", "--host", "127.0.0.1", "--port", "8002"
    ])
    
    # Wait a moment for server to start
    time.sleep(3)
    
    # Create ngrok tunnel
    print("🌐 Creating ngrok tunnel...")
    public_url = ngrok.connect(8002)
    
    print(f"\n✅ Your FastAPI app is now public!")
    print(f"🔗 Public URL: {public_url}")
    print(f"📚 API Docs: {public_url}/docs")
    print(f"🏥 Frontend: {public_url}/")
    print("\n📍 Press Ctrl+C to stop...")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        process.terminate()
        ngrok.disconnect(public_url)

if __name__ == "__main__":
    start_with_ngrok()
