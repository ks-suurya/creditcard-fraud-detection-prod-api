# src/api/main.py
from fastapi import FastAPI, Depends, HTTPException
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from src.api.models import PredictRequest, PredictResponse, PredictBatchRequest, PredictBatchResponse, HealthResponse
from src.api.security import api_key_auth
from src.logger import get_logger
from src.utils import get_env_var
from src.inference_realtime import predict_transaction
from src.inference_batch import invoke_batch_from_dataframe
import os

logger = get_logger("fraud-api")
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

ENDPOINT = get_env_var("SAGEMAKER_ENDPOINT", required=True)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check API readiness (basic checks only)."""
    try:
        # quick checks: env var presence and optional AWS creds presence
        region = os.getenv("AWS_REGION") or "not-set"
        cred = None
        try:
            import boto3
            cred = boto3.Session().get_credentials()
        except Exception:
            cred = None
        cred_ok = bool(cred and cred.get_frozen_credentials().access_key)
        if not cred_ok:
            logger.warning("AWS credentials not available inside runtime; ensure task role or credentials are set.")
        return HealthResponse(status="ok", endpoint=ENDPOINT)
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(api_key_auth)])
def predict(req: PredictRequest) -> PredictResponse:
    # Validate (pydantic ensures 30 floats) but double-check
    if len(req.features) != 30:
        raise HTTPException(status_code=400, detail=f"Invalid number of features: expected 30, got {len(req.features)}")

    try:
        score = predict_transaction(req.features)
        label = 1 if score >= 0.5 else 0
        return PredictResponse(probability=score, label=label)
    except NoCredentialsError:
        logger.exception("AWS credentials missing")
        raise HTTPException(status_code=500, detail="AWS credentials not found in runtime (task role or env).")
    except (ClientError, EndpointConnectionError) as e:
        logger.exception("Model invocation failed: %s", e)
        # If e contains response dict, we may have logged it already; return generic upstream error
        raise HTTPException(status_code=502, detail="Upstream model error")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Server error")

@app.post("/predict/batch", response_model=PredictBatchResponse, dependencies=[Depends(api_key_auth)])
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    for idx, vec in enumerate(req.transactions):
        if len(vec) != 30:
            raise HTTPException(status_code=400, detail=f"Transaction {idx} invalid feature count: expected 30, got {len(vec)}")
    try:
        probs = [predict_transaction(list(vec)) for vec in req.transactions]
        labels = [1 if p >= 0.5 else 0 for p in probs]
        return PredictBatchResponse(probabilities=probs, labels=labels)
    except NoCredentialsError:
        logger.exception("AWS credentials missing")
        raise HTTPException(status_code=500, detail="AWS credentials not found in runtime (task role or env).")
    except (ClientError, EndpointConnectionError):
        logger.exception("Model invocation failed")
        raise HTTPException(status_code=502, detail="Upstream model error")
    except Exception:
        logger.exception("Unexpected error in batch")
        raise HTTPException(status_code=500, detail="Server error")
