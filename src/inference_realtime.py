# src/inference_realtime.py
from typing import List
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from src.utils import get_env_var
from src.logger import get_logger
from src.preprocessing import Preprocessor

logger = get_logger(__name__)

REGION = get_env_var("AWS_REGION", "ap-south-1")
ENDPOINT = get_env_var("SAGEMAKER_ENDPOINT", required=True)
INFERENCE_COMPONENT = get_env_var("SAGEMAKER_INFERENCE", "")

# Preprocessor is loaded on import (cold-start)
_preprocessor = Preprocessor()
_preprocessor.load()

# boto client with retries
boto_config = Config(retries={"max_attempts": 3, "mode": "standard"})
_sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION, config=boto_config)


def predict_transaction(raw_features: List[float], threshold: float = 0.5) -> float:
    """
    Predicts fraud probability for a single transaction.
    raw_features: list of 30 numeric values [Time, V1..V28, Amount]
    returns probability float (0..1)
    """
    if not isinstance(raw_features, (list, tuple)):
        raise ValueError("features must be a list of numeric values")
    if len(raw_features) != 30:
        raise ValueError("Exactly 30 features required: [Time, V1..V28, Amount]")

    # Preprocess (scaling)
    features = _preprocessor.transform_vector(raw_features)

    payload = ",".join(map(str, features))
    kwargs = {"EndpointName": ENDPOINT, "ContentType": "text/csv", "Body": payload}
    if INFERENCE_COMPONENT:
        kwargs["InferenceComponentName"] = INFERENCE_COMPONENT

    try:
        resp = _sm_runtime.invoke_endpoint(**kwargs)
        score = float(resp["Body"].read().decode("utf-8").strip())
        logger.debug("predict_transaction -> %s", score)
        return score
    except (ClientError, EndpointConnectionError) as e:
        logger.exception("SageMaker invocation failed")
        raise
