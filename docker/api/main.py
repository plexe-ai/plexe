import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import routes - use absolute imports from project root
from api.app.routes import models, predictions, jobs

# Create FastAPI app with redirect_slashes=False
app = FastAPI(
    title="plexe API",
    description="API for managing ML models using plexe",
    version="0.1.0",
    redirect_slashes=False,
)

# Setup CORS
origins = os.getenv("CORS_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
