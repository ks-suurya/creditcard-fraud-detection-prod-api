# src/inference_batch.py
from typing import List, Optional
import pandas as pd
from src.logger import get_logger
from src.inference_realtime import predict_transaction
from src.preprocessing import Preprocessor

logger = get_logger(__name__)

# preprocessor is loaded in realtime module already; create fresh for batch if needed
_preprocessor = Preprocessor()
try:
    _preprocessor.load()
except Exception:
    logger.exception("Preprocessor load failed in batch module; continuing (may be degraded).")


def invoke_batch_from_dataframe(df: pd.DataFrame) -> List[Optional[float]]:
    """
    Score a dataframe of transactions and return list of probabilities (None for failed rows).
    Accepts DataFrame with either 30 columns (features) or 31 with 'Class' label in first column.
    """
    # Normalize dataframe to only feature columns
    if "Class" in df.columns:
        features_df = df.drop(columns=["Class"])
    elif df.shape[1] == len(_preprocessor.feature_order) + 1:
        # maybe label included as first column
        features_df = df.iloc[:, 1:]
    else:
        features_df = df.copy()

    try:
        arr = _preprocessor.transform_dataframe(features_df)
    except Exception:
        logger.exception("Failed to transform dataframe; converting to raw numpy array.")
        arr = features_df.values.astype(float)

    results = []
    for i, row in enumerate(arr):
        try:
            vec = list(map(float, row.tolist()))
            score = predict_transaction(vec)
            results.append(score)
            logger.debug("Row %s -> %s", i, score)
        except Exception as e:
            logger.exception("Row %s failed: %s", i, e)
            results.append(None)
    return results


def invoke_batch_from_csv(csv_path: str) -> List[Optional[float]]:
    logger.info("Loading CSV for batch inference: %s", csv_path)
    df = pd.read_csv(csv_path)
    return invoke_batch_from_dataframe(df)
