# src/handler.py
import json
from src.logger import get_logger
from src.utils import get_env_var
from src.inference_realtime import predict_transaction
from src.inference_batch import invoke_batch_from_dataframe
import base64
import pandas as pd

logger = get_logger(__name__)

def lambda_handler(event, context):
    """
    Minimal Lambda handler supporting:
     - API Gateway direct proxy for single transaction: pass JSON {"features": [ ... ]}
     - Batch from S3: not implemented here (could implement reading S3 path in event).
    """
    try:
        body = event.get("body") or event
        if isinstance(body, str):
            body = json.loads(body)
        # Single transaction
        if "features" in body:
            score = predict_transaction(body["features"])
            label = 1 if score >= 0.5 else 0
            return {
                "statusCode": 200,
                "body": json.dumps({"probability": score, "label": label}),
                "headers": {"Content-Type": "application/json"}
            }
        # Batch: accept list of transactions
        if "transactions" in body:
            probs = [predict_transaction(tx) for tx in body["transactions"]]
            labels = [1 if p >= 0.5 else 0 for p in probs]
            return {
                "statusCode": 200,
                "body": json.dumps({"probabilities": probs, "labels": labels}),
                "headers": {"Content-Type": "application/json"}
            }
        return {"statusCode": 400, "body": json.dumps({"error": "Missing 'features' or 'transactions'"})}
    except Exception as e:
        logger.exception("Lambda handler error")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
