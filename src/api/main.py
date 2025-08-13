# src/api/main.py
from fastapi import FastAPI, Depends, HTTPException
from botocore.exceptions import ClientError, EndpointConnectionError
from typing import List
from src.api.models import PredictRequest, PredictResponse, PredictBatchRequest, PredictBatchResponse, HealthResponse
from src.api.security import api_key_auth
from src.logger import get_logger
from src.utils import get_env_var
from src.inference_realtime import predict_transaction
from src.inference_batch import invoke_batch_from_dataframe

logger = get_logger("fraud-api")
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

# Simple CORS (tighten in production)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["POST","GET"], allow_headers=["*"])

ENDPOINT = get_env_var("SAGEMAKER_ENDPOINT", required=True)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", endpoint=ENDPOINT)

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(api_key_auth)])
def predict(req: PredictRequest) -> PredictResponse:
    try:
        score = predict_transaction(req.features)
        label = 1 if score >= 0.5 else 0
        return PredictResponse(probability=score, label=label)
    except (ClientError, EndpointConnectionError) as e:
        logger.exception("Model invocation failed")
        raise HTTPException(status_code=502, detail="Upstream model error")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Server error")

@app.post("/predict/batch", response_model=PredictBatchResponse, dependencies=[Depends(api_key_auth)])
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    try:
        probs = [predict_transaction(list(vec)) for vec in req.transactions]
        labels = [1 if p >= 0.5 else 0 for p in probs]
        return PredictBatchResponse(probabilities=probs, labels=labels)
    except (ClientError, EndpointConnectionError):
        logger.exception("Model invocation failed")
        raise HTTPException(status_code=502, detail="Upstream model error")
    except Exception:
        logger.exception("Unexpected error in batch")
        raise HTTPException(status_code=500, detail="Server error")
