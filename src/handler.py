import os
import json
import base64
import io
import boto3
import joblib
import pandas as pd
from src.preprocessing import preprocess
from logger import log_info, log_error
from src.api.security import check_api_key

# Environment variables
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
S3_BUCKET = os.getenv("S3_BUCKET")
SCALER_KEY = os.getenv("SCALER_KEY")
FEATURE_ORDER_KEY = os.getenv("FEATURE_ORDER_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# Artifacts cache
scaler = None
feature_order = None

def load_artifacts():
    global scaler, feature_order
    if scaler is None:
        log_info("Loading scaler from S3", key=SCALER_KEY)
        scaler_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=SCALER_KEY)
        scaler = joblib.load(io.BytesIO(scaler_obj["Body"].read()))
    if feature_order is None:
        log_info("Loading feature order from S3", key=FEATURE_ORDER_KEY)
        feature_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=FEATURE_ORDER_KEY)
        feature_order = json.loads(feature_obj["Body"].read().decode("utf-8"))

load_artifacts()

def lambda_handler(event, context):
    try:
        log_info("Received event", event_id=context.aws_request_id)

        # --- Security Check ---
        if not check_api_key(event.get("headers", {})):
            log_error("Unauthorized request", event_id=context.aws_request_id)
            return {"statusCode": 403, "body": json.dumps({"error": "Invalid API key"})}

        # --- Parse Body ---
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)

        predictions = []

        # --- Real-time JSON mode ---
        if "instances" in body:
            log_info("Processing JSON real-time request")
            processed = preprocess(body["instances"], scaler, feature_order)
            payload = "\n".join([",".join(map(str, row)) for row in processed])

        # --- Batch CSV mode ---
        elif "csv_base64" in body:
            log_info("Processing batch CSV request")
            csv_bytes = base64.b64decode(body["csv_base64"])
            df = pd.read_csv(io.BytesIO(csv_bytes))
            processed = preprocess(df.to_dict(orient="records"), scaler, feature_order)
            payload = "\n".join([",".join(map(str, row)) for row in processed])

        else:
            return {"statusCode": 400, "body": json.dumps({"error": "No valid data provided"})}

        # --- Invoke SageMaker ---
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )

        preds = [float(p) for p in response["Body"].read().decode("utf-8").strip().split("\n")]
        predictions.extend(preds)

        log_info("Inference complete", predictions=predictions)

        return {
            "statusCode": 200,
            "body": json.dumps({"predictions": predictions})
        }

    except Exception as e:
        log_error("Unhandled error", error=str(e))
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
