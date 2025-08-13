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
    Loads scaler and feature_order from local artifacts/ or S3, and transforms vectors/dataframes.
    Assumes the trained scaler expects the full feature vector (30 features).
    """
    def __init__(self,
                 local_artifacts_dir: str = "artifacts",
                 s3_bucket: Optional[str] = None,
                 scaler_key: Optional[str] = None,
                 feature_order_key: Optional[str] = None):
        self.local_dir = local_artifacts_dir
        self.s3_bucket = s3_bucket or get_env_var("S3_BUCKET", "", required=False)
        self.scaler_key = scaler_key or get_env_var("SCALER_KEY", "artifacts/scaler.joblib", required=False)
        self.feature_order_key = feature_order_key or get_env_var("FEATURE_ORDER_KEY", "artifacts/feature_order.json", required=False)
        self.scaler = None
        self.feature_order = None
        try:
            self.s3 = boto3.client("s3", region_name=get_env_var("AWS_REGION", "ap-south-1", required=False))
        except Exception:
            self.s3 = None

    def load(self):
        """
        Try local artifacts first, then S3 if configured. Errors are logged but not raised so service can still run (with fallback behavior).
        """
        # Local paths (artifact filenames)
        scaler_local = os.path.join(self.local_dir, os.path.basename(self.scaler_key or "scaler.joblib"))
        feature_local = os.path.join(self.local_dir, os.path.basename(self.feature_order_key or "feature_order.json"))

        # Load scaler locally if present
        if os.path.exists(scaler_local):
            try:
                self.scaler = joblib.load(scaler_local)
                logger.info("Loaded scaler from local artifacts.")
            except Exception:
                logger.exception("Failed to load local scaler; will try S3 if configured.")
                self.scaler = None

        # Load feature_order locally if present
        if os.path.exists(feature_local):
            try:
                with open(feature_local, "r") as f:
                    self.feature_order = json.load(f)
                logger.info("Loaded feature_order from local artifacts.")
            except Exception:
                logger.exception("Failed to load local feature_order; will try S3 if configured.")
                self.feature_order = None

        # Fallback: load from S3 if bucket provided and missing locally
        if (self.scaler is None or self.feature_order is None) and self.s3_bucket and self.s3:
            try:
                if self.scaler is None and self.scaler_key:
                    resp = self.s3.get_object(Bucket=self.s3_bucket, Key=self.scaler_key)
                    self.scaler = joblib.load(io.BytesIO(resp["Body"].read()))
                    logger.info(f"Loaded scaler from s3://{self.s3_bucket}/{self.scaler_key}")
            except ClientError as e:
                logger.warning("Could not load scaler from S3: %s", e)
            except Exception:
                logger.exception("Error loading scaler from S3.")

            try:
                if self.feature_order is None and self.feature_order_key:
                    resp = self.s3.get_object(Bucket=self.s3_bucket, Key=self.feature_order_key)
                    self.feature_order = json.loads(resp["Body"].read().decode("utf-8"))
                    logger.info(f"Loaded feature_order from s3://{self.s3_bucket}/{self.feature_order_key}")
            except ClientError as e:
                logger.warning("Could not load feature_order from S3: %s", e)
            except Exception:
                logger.exception("Error loading feature_order from S3.")

        # If still missing, populate a sensible default feature order (Time, V1..V28, Amount)
        if self.feature_order is None:
            self.feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            logger.warning("feature_order not found; using default Time + V1..V28 + Amount")

        logger.debug("Preprocessor ready. scaler=%s feature_order_len=%d", bool(self.scaler), len(self.feature_order))

    def transform_vector(self, vec: List[float]) -> List[float]:
        """
        Accept a single transaction vector of length 30 and return the transformed vector.
        If scaler is present, it will transform the full 30-feature vector.
        """
        if not isinstance(vec, (list, tuple, np.ndarray)):
            raise ValueError("Input must be a list/tuple/ndarray of numeric features.")
        if len(vec) != len(self.feature_order):
            raise ValueError(f"Expected {len(self.feature_order)} features (got {len(vec)}).")

        arr = np.array(vec, dtype=float).reshape(1, -1)
        if self.scaler is None:
            # scaler not available — return original vector (but caller should be aware)
            logger.warning("Scaler not loaded; returning raw features.")
            return arr.flatten().tolist()

        try:
            out = self.scaler.transform(arr)
            return out.flatten().tolist()
        except Exception:
            logger.exception("Failed to apply scaler; returning original vector")
            return arr.flatten().tolist()

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert a dataframe to a numeric 2D array in the correct order and apply scaler if present.
        """
        # If DataFrame has header names matching feature_order, reorder; else try to pick first 30 numeric cols
        try:
            if list(df.columns) == self.feature_order:
                df_ordered = df[self.feature_order]
            else:
                # Try to pick columns by name where possible
                common = [c for c in self.feature_order if c in df.columns]
                if len(common) == len(self.feature_order):
                    df_ordered = df[self.feature_order]
                else:
                    # fallback to numeric first N columns (assume label may be first)
                    if df.shape[1] == len(self.feature_order) + 1 and "Class" in df.columns:
                        df_ordered = df.drop(columns=["Class"]).iloc[:, :len(self.feature_order)]
                    else:
                        df_ordered = df.iloc[:, :len(self.feature_order)]
        except Exception:
            df_ordered = df.iloc[:, :len(self.feature_order)]

        arr = df_ordered.values.astype(float)
        if self.scaler is None:
            return arr

        try:
            arr_t = self.scaler.transform(arr)
            return arr_t
        except Exception:
            logger.exception("Failed to apply scaler to dataframe; returning raw array")
            return arr
