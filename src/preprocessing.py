# src/preprocessing.py
import io
import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError

from src.logger import get_logger
from src.utils import get_env_var

logger = get_logger(__name__)

class Preprocessor:
    """
    Loads scaler and feature_order (local artifacts/ or S3 fallback).
    Exposes transform_vector and transform_dataframe which return numeric arrays.
    """

    def __init__(self,
                 local_artifacts_dir: str = "artifacts",
                 s3_bucket: Optional[str] = None,
                 scaler_key: Optional[str] = None,
                 feature_order_key: Optional[str] = None):
        self.local_artifacts_dir = local_artifacts_dir
        self.s3_bucket = s3_bucket or get_env_var("S3_BUCKET", "", required=False)
        self.scaler_key = scaler_key or get_env_var("SCALER_KEY", "artifacts/scaler.joblib", required=False)
        self.feature_order_key = feature_order_key or get_env_var("FEATURE_ORDER_KEY", "artifacts/feature_order.json", required=False)
        self.scaler = None
        self.feature_order = None

        # boto3 client for S3 (if needed)
        try:
            region = get_env_var("AWS_REGION", required=False) or None
            self.s3 = boto3.client("s3", region_name=region) if self.s3_bucket else None
        except Exception:
            self.s3 = None

    def load(self):
        """Load artifacts: prefer local files, fallback to S3 (if configured)."""
        # local file paths
        scaler_local = os.path.join(self.local_artifacts_dir, os.path.basename(self.scaler_key or "scaler.joblib"))
        feature_local = os.path.join(self.local_artifacts_dir, os.path.basename(self.feature_order_key or "feature_order.json"))

        # load scaler locally
        if os.path.exists(scaler_local):
            try:
                self.scaler = joblib.load(scaler_local)
                logger.info("Loaded scaler from local artifacts: %s", scaler_local)
            except Exception:
                logger.exception("Failed to load local scaler; will try S3 fallback.")
                self.scaler = None

        # load feature_order locally
        if os.path.exists(feature_local):
            try:
                with open(feature_local, "r") as f:
                    self.feature_order = json.load(f)
                logger.info("Loaded feature_order from local artifacts: %s", feature_local)
            except Exception:
                logger.exception("Failed to load local feature_order; will try S3 fallback.")
                self.feature_order = None

        # fallback: load from S3 if configured
        if (self.scaler is None or self.feature_order is None) and self.s3 and self.s3_bucket:
            try:
                if self.scaler is None and self.scaler_key:
                    obj = self.s3.get_object(Bucket=self.s3_bucket, Key=self.scaler_key)
                    self.scaler = joblib.load(io.BytesIO(obj["Body"].read()))
                    logger.info("Loaded scaler from s3://%s/%s", self.s3_bucket, self.scaler_key)
            except ClientError as e:
                logger.warning("S3 scaler load failed: %s", e)
            except Exception:
                logger.exception("Unexpected error loading scaler from S3.")

            try:
                if self.feature_order is None and self.feature_order_key:
                    obj = self.s3.get_object(Bucket=self.s3_bucket, Key=self.feature_order_key)
                    self.feature_order = json.loads(obj["Body"].read().decode("utf-8"))
                    logger.info("Loaded feature_order from s3://%s/%s", self.s3_bucket, self.feature_order_key)
            except ClientError as e:
                logger.warning("S3 feature_order load failed: %s", e)
            except Exception:
                logger.exception("Unexpected error loading feature_order from S3.")

        # final fallback: assume standard creditcard dataset order if missing
        if self.feature_order is None:
            self.feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            logger.warning("feature_order not found; using default Time + V1..V28 + Amount")

        # If scaler exists, and it reports n_features_in_, ensure sizes align (log warning otherwise)
        if self.scaler is not None:
            try:
                n_in = getattr(self.scaler, "n_features_in_", None)
                if n_in is not None and n_in != len(self.feature_order):
                    logger.warning("Scaler expects %s features but feature_order length is %s. Using scaler.n_features_in_=%s",
                                   n_in, len(self.feature_order), n_in)
                    # If scaler expects different length, do not silently reshape — we'll rely on scaler behavior later
            except Exception:
                logger.exception("Error checking scaler n_features_in_")

        logger.debug("Preprocessor ready. scaler=%s, feature_order_len=%d", bool(self.scaler), len(self.feature_order))

    def transform_vector(self, vec: List[float]) -> List[float]:
        """Transform a single vector. Must be the same length as feature_order."""
        if not isinstance(vec, (list, tuple, np.ndarray)):
            raise ValueError("Input features must be a list/tuple/ndarray of numbers.")

        if len(vec) != len(self.feature_order):
            raise ValueError(f"Feature length mismatch: expected {len(self.feature_order)}, got {len(vec)}")

        arr = np.array(vec, dtype=float).reshape(1, -1)

        if self.scaler is None:
            logger.warning("Scaler not loaded; returning raw numeric vector.")
            return arr.flatten().tolist()

        try:
            out = self.scaler.transform(arr)
            return out.flatten().tolist()
        except Exception:
            logger.exception("Scaler.transform failed; returning raw numeric vector.")
            return arr.flatten().tolist()

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a dataframe into a 2D numpy array aligned to feature_order:
        - If df columns match feature_order -> reorder
        - Else, try to pick columns by name
        - Else take first N numeric columns (N = len(feature_order))
        """
        target_cols = self.feature_order
        try:
            if list(df.columns) == target_cols:
                df_ordered = df[target_cols]
            else:
                # if all target cols exist in df, pick those
                if all(c in df.columns for c in target_cols):
                    df_ordered = df[target_cols]
                else:
                    # if df has 'Class' label column, drop it
                    if "Class" in df.columns and df.shape[1] >= len(target_cols) + 1:
                        df_ordered = df.drop(columns=["Class"]).iloc[:, :len(target_cols)]
                    else:
                        df_ordered = df.iloc[:, :len(target_cols)]
        except Exception:
            df_ordered = df.iloc[:, :len(target_cols)]

        arr = df_ordered.values.astype(float)
        if self.scaler is None:
            return arr

        try:
            arr_t = self.scaler.transform(arr)
            return arr_t
        except Exception:
            logger.exception("Scaler.transform on dataframe failed; returning raw array")
            return arr
