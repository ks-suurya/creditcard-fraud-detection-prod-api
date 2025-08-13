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
ENDPOINT = get_env_var("SAGEMAKER_ENDPOINT", None, required=True)
INFERENCE_COMPONENT = get_env_var("SAGEMAKER_INFERENCE", "", required=False)

# instantiate preprocessor and load artifacts (local first, S3 fallback)
_preprocessor = Preprocessor()
try:
    _preprocessor.load()
except Exception:
    logger.exception("Preprocessor load had an exception; continuing with possible degraded behavior.")

# boto client with retries
boto_config = Config(retries={"max_attempts": 3, "mode": "standard"})
_sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION, config=boto_config)


def predict_transaction(raw_features: List[float]) -> float:
    """
    Real-time prediction for a single transaction.
    - Validate length (must equal preprocessor.feature_order length)
    - Apply preprocessing (scaler)
    - Call SageMaker endpoint with CSV payload (text/csv)
    Returns float probability.
    Raises NoCredentialsError if AWS creds not present.
    """
    if not isinstance(raw_features, (list, tuple)):
        raise ValueError("features must be a list of numeric values")

    expected_len = len(_preprocessor.feature_order)
    if len(raw_features) != expected_len:
        raise ValueError(f"Expected {expected_len} features, got {len(raw_features)}")

    # Preprocess (applies scaler if available)
    features = _preprocessor.transform_vector(raw_features)

    # Build CSV payload (single row)
    payload = ",".join(map(str, features))

    kwargs = {"EndpointName": ENDPOINT, "ContentType": "text/csv", "Body": payload}
    if INFERENCE_COMPONENT:
        kwargs["InferenceComponentName"] = INFERENCE_COMPONENT

    try:
        resp = _sm_runtime.invoke_endpoint(**kwargs)
        score = float(resp["Body"].read().decode("utf-8").strip())
        logger.debug("predict_transaction -> %s", score)
        return score
    except NoCredentialsError:
        logger.exception("AWS credentials missing (NoCredentialsError)")
        raise
    except (ClientError, EndpointConnectionError):
        logger.exception("SageMaker invocation failed")
        raise
