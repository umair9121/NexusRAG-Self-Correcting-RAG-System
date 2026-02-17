"""
Run the FastAPI backend from any directory.
Ensures the backend folder is on the path so "app" resolves correctly.
"""
import os
import sys

# Run from backend/ so that "app" is backend.app and all routes load
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
