# src/api/main.py
from fastapi import FastAPI, Depends, HTTPException
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from typing import List
from src.api.models import PredictRequest, PredictResponse, PredictBatchRequest, PredictBatchResponse, HealthResponse
from src.api.security import api_key_auth
from src.logger import get_logger
from src.utils import get_env_var
from src.inference_realtime import predict_transaction
from src.inference_batch import invoke_batch_from_dataframe
import os

logger = get_logger("fraud-api")
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

# Simple CORS (tighten in production)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

ENDPOINT = get_env_var("SAGEMAKER_ENDPOINT", required=True)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check API readiness and basic AWS connectivity."""
    try:
        region = os.getenv("AWS_REGION") or "not-set"
        cred_file_exists = os.path.exists(os.path.expanduser("~/.aws/credentials"))
        if not cred_file_exists:
            logger.warning("AWS credentials file not found inside runtime (check mount or env variables).")
        return HealthResponse(status="ok", endpoint=ENDPOINT)
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(api_key_auth)])
def predict(req: PredictRequest) -> PredictResponse:
    """Single transaction prediction."""
    # Validate feature length immediately
    if len(req.features) != 30:
        raise HTTPException(status_code=400, detail=f"Invalid number of features: expected 30, got {len(req.features)}")

    try:
        score = predict_transaction(req.features)
        label = 1 if score >= 0.5 else 0
        return PredictResponse(probability=score, label=label)
    except NoCredentialsError:
        logger.exception("AWS credentials missing")
        raise HTTPException(status_code=500, detail="AWS credentials not found in runtime (mount or env).")
    except (ClientError, EndpointConnectionError):
        logger.exception("Model invocation failed")
        raise HTTPException(status_code=502, detail="Upstream model error")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Server error")

@app.post("/predict/batch", response_model=PredictBatchResponse, dependencies=[Depends(api_key_auth)])
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """Batch transaction prediction (transactions is a list of 30-length vectors)."""
    for idx, vec in enumerate(req.transactions):
        if len(vec) != 30:
            raise HTTPException(status_code=400, detail=f"Transaction {idx} invalid feature count: expected 30, got {len(vec)}")

    try:
        probs = [predict_transaction(list(vec)) for vec in req.transactions]
        labels = [1 if p >= 0.5 else 0 for p in probs]
        return PredictBatchResponse(probabilities=probs, labels=labels)
    except NoCredentialsError:
        logger.exception("AWS credentials missing")
        raise HTTPException(status_code=500, detail="AWS credentials not found in runtime (mount or env).")
    except (ClientError, EndpointConnectionError):
        logger.exception("Model invocation failed")
        raise HTTPException(status_code=502, detail="Upstream model error")
    except Exception:
        logger.exception("Unexpected error in batch")
        raise HTTPException(status_code=500, detail="Server error")
