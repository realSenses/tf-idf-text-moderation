from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
from inference import ToxicCommentClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Toxic Comment Moderation API",
    description="API for detecting toxic comments using multi-label classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None


class CommentRequest(BaseModel):
    text: str = Field(..., description="The comment text to analyze", min_length=1, max_length=5000)


class BatchCommentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of comment texts to analyze", min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    text: str
    is_toxic: bool
    toxic_categories: List[str]
    predictions: Dict[str, bool]
    probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    global classifier
    try:
        logger.info("Loading model...")
        classifier = ToxicCommentClassifier()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Toxic Comment Moderation API",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy" if classifier is not None else "unhealthy",
        "model_loaded": classifier is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: CommentRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = classifier.predict(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchCommentRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = classifier.predict_batch(request.texts)
        return BatchPredictionResponse(results=[PredictionResponse(**r) for r in results])
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/categories", response_model=Dict[str, List[str]])
async def get_categories():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"categories": classifier.toxic_categories}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)