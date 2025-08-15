# src/inference_realtime.py
from typing import List
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from src.utils import get_env_var
from src.logger import get_logger
from src.preprocessing import Preprocessor

logger = get_logger(__name__)

REGION = get_env_var("AWS_REGION", "ap-south-1")
ENDPOINT = get_env_var("SAGEMAKER_ENDPOINT", required=True)
INFERENCE_COMPONENT = get_env_var("SAGEMAKER_INFERENCE", "", required=False)

# Preprocessor singleton
_preprocessor = Preprocessor()
try:
    _preprocessor.load()
except Exception:
    logger.exception("Preprocessor load failed at module init; continuing (degraded).")

# boto3 sagemaker-runtime client with retry config
boto_cfg = Config(retries={"max_attempts": 3, "mode": "standard"})
_sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION, config=boto_cfg)


def predict_transaction(raw_features: List[float]) -> float:
    """
    Validate -> preprocess -> invoke SageMaker endpoint -> return float probability.
    Raises ValueError for bad input; raises NoCredentialsError if AWS creds missing.
    """
    if not isinstance(raw_features, (list, tuple)):
        raise ValueError("features must be a list of numeric values")

    expected_len = len(_preprocessor.feature_order)
    if len(raw_features) != expected_len:
        raise ValueError(f"Expected {expected_len} features, got {len(raw_features)}")

    # Apply preprocessing (scaling) if scaler available
    features = _preprocessor.transform_vector(raw_features)

    # Build CSV payload string (one row, comma-separated, no header)
    # Use simple string conversion to preserve numeric format acceptable to XGBoost container
    payload = ",".join(map(str, features))
    content_type = "text/csv"

    kwargs = {
        "EndpointName": ENDPOINT,
        "ContentType": content_type,
        "Body": payload
    }
    if INFERENCE_COMPONENT:
        kwargs["InferenceComponentName"] = INFERENCE_COMPONENT

    # Debug logging: show the exact payload and content type (temporary / helpful)
    logger.debug("Invoking SageMaker endpoint '%s' with payload: %s", ENDPOINT, payload)
    logger.debug("SageMaker invoke kwargs: %s", {k: (v if k != "Body" else "<body...>") for k, v in kwargs.items()})

    try:
        resp = _sm_runtime.invoke_endpoint(**kwargs)
        body = resp["Body"].read().decode("utf-8").strip()
        logger.debug("Raw SageMaker response body: %s", body)
        # Model often returns a single float as string
        try:
            score = float(body)
            return score
        except ValueError:
            # sometimes model returns JSON; try to parse JSON float inside
            import json
            try:
                parsed = json.loads(body)
                # If structure is {"predictions":[x]} or [x]
                if isinstance(parsed, dict) and "predictions" in parsed:
                    val = parsed["predictions"][0]
                    return float(val)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return float(parsed[0])
            except Exception:
                logger.exception("Unable to parse SageMaker response body into float")
                raise RuntimeError("Unable to parse model response")
    except NoCredentialsError:
        logger.exception("AWS credentials missing (NoCredentialsError)")
        raise
    except ClientError as e:
        # Log entire response returned by SageMaker container (e.response contains details)
        logger.exception("SageMaker ClientError: %s", getattr(e, "response", str(e)))
        # Re-raise so calling layer (FastAPI) can convert to HTTP 502
        raise
    except EndpointConnectionError:
        logger.exception("EndpointConnectionError when calling SageMaker")
        raise
    except Exception:
        logger.exception("Unexpected error during SageMaker invocation")
        raise
