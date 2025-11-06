#!/usr/bin/env python3
"""
Script to run the Personal Finance Chatbot application
Handles both backend and frontend startup
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import fastapi
        import uvicorn
        import requests
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Please create a .env file with your IBM Watson credentials")
        print("See env_example.txt for reference")
        return False
    print("✅ .env file found")
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        # Start backend in a subprocess
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait longer for the server to start (models need time to load)
        print("⏳ Waiting for models to load (this may take 30-60 seconds)...")
        
        # Check for startup messages
        startup_success = False
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            
            # Check if process is still running
            if backend_process.poll() is not None:
                # Process died, get error output
                stdout, stderr = backend_process.communicate()
                print(f"❌ Backend process failed to start:")
                if stderr:
                    print(f"Error: {stderr}")
                return None
            
            # Check if server is responding
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend server is running on http://localhost:8000")
                    startup_success = True
                    break
            except requests.exceptions.RequestException:
                continue
        
        if not startup_success:
            print("❌ Backend server failed to start within 30 seconds")
            backend_process.terminate()
            return None
        
        return backend_process
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    try:
        # Start frontend in a subprocess
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501", "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        print("✅ Frontend is starting on http://localhost:8501")
        print("⏳ Please wait a moment for the browser to open...")
        
        return frontend_process
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def cleanup(backend_process, frontend_process):
    """Clean up processes on exit"""
    print("\n🛑 Shutting down servers...")
    
    if backend_process:
        backend_process.terminate()
        print("✅ Backend server stopped")
    
    if frontend_process:
        frontend_process.terminate()
        print("✅ Frontend server stopped")

def main():
    """Main function to run the application"""
    print("=" * 50)
    print("💰 Personal Finance Chatbot")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        print("⚠️  Continuing without .env file (using fallback responses)")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting...")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Stopping backend...")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Application is running!")
    print("=" * 50)
    print("📱 Frontend: http://localhost:8501")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        cleanup(backend_process, frontend_process)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep the main process running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    finally:
        cleanup(backend_process, frontend_process)

if __name__ == "__main__":
    main()
