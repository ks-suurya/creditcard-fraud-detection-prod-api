from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from typing import List

from src.api.models import (
    PredictRequest, PredictResponse,
    PredictBatchRequest, PredictBatchResponse,
    HealthResponse
)
from src.api.security import api_key_auth
from src.utils import get_env_var
from src.logger import get_logger

logger = get_logger("fraud-api")

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

# CORS (adjust for your domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# --- AWS / SageMaker clients (module scope, reuse across requests) ---
REGION = get_env_var("AWS_REGION")
ENDPOINT = get_env_var("SAGEMAKER_ENDPOINT")
INFERENCE_COMPONENT = get_env_var("SAGEMAKER_INFERENCE", None)

boto_config = Config(retries={"max_attempts": 3, "mode": "standard"})
sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION, config=boto_config)

# --- Helpers ---
def invoke_sagemaker_csv_row(values: List[float]) -> float:
    payload = ",".join(map(str, values))
    kwargs = dict(
        EndpointName=ENDPOINT,
        ContentType="text/csv",
        Body=payload
    )
    # Only pass InferenceComponentName if provided
    if INFERENCE_COMPONENT:
        kwargs["InferenceComponentName"] = INFERENCE_COMPONENT

    resp = sm_runtime.invoke_endpoint(**kwargs)
    score = float(resp["Body"].read().decode("utf-8").strip())
    return score

# --- Routes ---
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", endpoint=ENDPOINT)

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(api_key_auth)])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Real-time prediction for a single transaction.
    Body: {"features": [Time, V1..V28, Amount]}
    """
    try:
        score = invoke_sagemaker_csv_row(req.features)
        label = 1 if score >= 0.5 else 0
        return PredictResponse(probability=score, label=label)
    except (ClientError, EndpointConnectionError) as e:
        logger.error(f"SageMaker error: {e}")
        raise HTTPException(status_code=502, detail="Upstream model error")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Server error")

@app.post("/predict/batch", response_model=PredictBatchResponse, dependencies=[Depends(api_key_auth)])
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    """
    Batch prediction.
    Body: {"transactions": [[...30 vals...], [...], ...]}
    """
    probs: List[float] = []
    try:
        for row in req.transactions:
            score = invoke_sagemaker_csv_row(row)
            probs.append(score)
        labels = [1 if p >= 0.5 else 0 for p in probs]
        return PredictBatchResponse(probabilities=probs, labels=labels)
    except (ClientError, EndpointConnectionError) as e:
        logger.error(f"SageMaker error: {e}")
        raise HTTPException(status_code=502, detail="Upstream model error")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Server error")
