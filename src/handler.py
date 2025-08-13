# src/handler.py
import json
from src.logger import get_logger
from src.inference_realtime import predict_transaction
from src.inference_batch import invoke_batch_from_dataframe

logger = get_logger(__name__)

def lambda_handler(event, context):
    """
    Support API Gateway proxy for single transaction ({"features":[...]})
    or batch ({"transactions":[ [...], [...] ]}).
    """
    try:
        body = event.get("body") or event
        if isinstance(body, str):
            body = json.loads(body)

        if "features" in body:
            score = predict_transaction(body["features"])
            label = 1 if score >= 0.5 else 0
            return {"statusCode": 200, "body": json.dumps({"probability": score, "label": label})}

        if "transactions" in body:
            probs = [predict_transaction(tx) for tx in body["transactions"]]
            labels = [1 if p >= 0.5 else 0 for p in probs]
            return {"statusCode": 200, "body": json.dumps({"probabilities": probs, "labels": labels})}

        return {"statusCode": 400, "body": json.dumps({"error": "Missing 'features' or 'transactions' in body"})}

    except Exception as e:
        logger.exception("Lambda handler error")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
