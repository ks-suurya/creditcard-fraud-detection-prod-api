# src/inference.py
import boto3
import pandas as pd
from src.utils import get_env_var

# Load config from .env
region = get_env_var("AWS_REGION")
endpoint_name = get_env_var("SAGEMAKER_ENDPOINT")
inference_name = get_env_var("SAGEMAKER_INFERENCE")

# SageMaker runtime client
sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

def invoke_endpoint(csv_path: str):
    df = pd.read_csv(csv_path)
    if 'Class' in df.columns:
        features = df.drop(columns=['Class'])
    else:
        # If no 'Class' column, assume the first column is the label (as in original code)
        features = df.iloc[:, 1:]  # Skip first column

    predictions = []
    for i, row in features.iterrows():
        payload = ",".join(map(str, row.tolist()))
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            InferenceComponentName=inference_name,
            ContentType="text/csv",
            Body=payload
        )
        result = float(response["Body"].read().decode("utf-8"))
        predictions.append(result)
        print(f"Row {i} prediction: {result}")

    return predictions

if __name__ == "__main__":
    test_csv = "../data/holdout.csv"  # Ensure this exists
    invoke_endpoint(test_csv)
