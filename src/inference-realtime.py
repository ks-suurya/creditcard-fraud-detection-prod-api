# src/inference_realtime.py
import boto3
from src.utils import get_env_var
from src.logger import get_logger

logger = get_logger(__name__)

region = get_env_var("AWS_REGION")
endpoint_name = get_env_var("SAGEMAKER_ENDPOINT")
inference_name = get_env_var("SAGEMAKER_INFERENCE", None)

sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

def predict_transaction(features: list[float]) -> float:
    """
    Real-time fraud prediction for a single transaction.
    :param features: List of numerical features.
    :return: Fraud probability (0-1)
    """
    payload = ",".join(map(str, features))
    logger.debug(f"Payload: {payload}")

    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        InferenceComponentName=inference_name if inference_name else None,
        ContentType="text/csv",
        Body=payload
    )
    result = float(response["Body"].read().decode("utf-8"))
    logger.info(f"Prediction: {result}")
    return result

if __name__ == "__main__":
    sample_transaction = [0.0, 1.2, -0.3, 200.5, 0.0, 1.0]  # Example
    score = predict_transaction(sample_transaction)
    logger.info(f"Fraud probability: {score}")
