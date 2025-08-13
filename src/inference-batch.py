# src/inference_batch.py
import boto3
import pandas as pd
from src.utils import get_env_var
from src.logger import get_logger

logger = get_logger(__name__)

region = get_env_var("AWS_REGION")
endpoint_name = get_env_var("SAGEMAKER_ENDPOINT")
inference_name = get_env_var("SAGEMAKER_INFERENCE", None)

sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

def invoke_batch(csv_path: str):
    logger.info(f"Starting batch inference for {csv_path}")
    df = pd.read_csv(csv_path)

    if 'Class' in df.columns:
        features = df.drop(columns=['Class'])
    else:
        features = df.iloc[:, 1:]

    predictions = []
    for i, row in features.iterrows():
        payload = ",".join(map(str, row.tolist()))
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            InferenceComponentName=inference_name if inference_name else None,
            ContentType="text/csv",
            Body=payload
        )
        result = float(response["Body"].read().decode("utf-8"))
        predictions.append(result)
        logger.debug(f"Row {i} prediction: {result}")

    logger.info("Batch inference complete.")
    return predictions

if __name__ == "__main__":
    test_csv = "data/holdout.csv"
    results = invoke_batch(test_csv)
    logger.info(f"Predictions: {results[:10]} ...")
