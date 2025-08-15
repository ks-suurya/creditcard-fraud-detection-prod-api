# src/inference_batch.py
from typing import List, Optional
import pandas as pd
from src.logger import get_logger
from src.inference_realtime import predict_transaction
from src.preprocessing import Preprocessor

logger = get_logger(__name__)

# The Preprocessor is already initialized in inference_realtime module, but create local one for safety
_pre = Preprocessor()
try:
    _pre.load()
except Exception:
    logger.exception("Preprocessor load failed in batch module; continuing with degraded behavior.")

def invoke_batch_from_dataframe(df: pd.DataFrame) -> List[Optional[float]]:
    """
    Transform dataframe into features and call predict_transaction for each row.
    Returns list of probabilities (or None for failed rows).
    """
    # Normalize to feature-only DataFrame
    if "Class" in df.columns:
        features_df = df.drop(columns=["Class"])
    elif df.shape[1] == len(_pre.feature_order) + 1:
        # if label present as first column
        features_df = df.iloc[:, 1:]
    else:
        features_df = df.copy()

    try:
        arr = _pre.transform_dataframe(features_df)
    except Exception:
        logger.exception("Failed to transform dataframe; falling back to raw values.")
        arr = features_df.values.astype(float)

    results = []
    for i, row in enumerate(arr):
        try:
            vec = list(map(float, row.tolist()))
            score = predict_transaction(vec)
            results.append(score)
        except Exception as e:
            logger.exception("Batch row %s failed: %s", i, e)
            results.append(None)
    return results

def invoke_batch_from_csv(csv_path: str) -> List[Optional[float]]:
    logger.info("Loading CSV for batch inference: %s", csv_path)
    df = pd.read_csv(csv_path)
    return invoke_batch_from_dataframe(df)
