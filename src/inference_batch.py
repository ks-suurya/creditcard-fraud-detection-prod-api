# src/inference_batch.py
from typing import List, Optional
import pandas as pd
from src.logger import get_logger
from src.inference_realtime import predict_transaction
from src.preprocessing import Preprocessor

logger = get_logger(__name__)

# Load preprocessor once
_preprocessor = Preprocessor()
_preprocessor.load()


def invoke_batch_from_dataframe(df: pd.DataFrame) -> List[Optional[float]]:
    """
    Accepts a DataFrame and returns list of probabilities (None for failed rows).
    Expects a DataFrame with either:
      - headered columns matching feature_order
      - or 31 columns (Class + 30 features) which will drop the first column
      - or exactly 30 feature columns
    """
    if "Class" in df.columns:
        features_df = df.drop(columns=["Class"])
    elif df.shape[1] == 31:
        features_df = df.iloc[:, 1:]
    else:
        features_df = df.copy()

    # ensure numeric and correct ordering if possible
    try:
        arr = _preprocessor.transform_dataframe(features_df)
    except Exception:
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
