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
    Loads preprocessing artifacts (scaler & feature order) from local artifacts/ or S3.
    Usage:
      p = Preprocessor()
      p.load()  # loads from local artifacts if available, otherwise from S3
      vec_scaled = p.transform_vector(vec)  # vec as list of 30 floats
    """

    def __init__(self,
                 local_artifacts_dir: str = "artifacts",
                 s3_bucket: Optional[str] = None,
                 scaler_key: Optional[str] = None,
                 feature_order_key: Optional[str] = None):
        self.local_dir = local_artifacts_dir
        self.s3_bucket = s3_bucket or get_env_var("S3_BUCKET", "")
        self.scaler_key = scaler_key or get_env_var("SCALER_KEY", "artifacts/scaler.joblib")
        self.feature_order_key = feature_order_key or get_env_var("FEATURE_ORDER_KEY", "artifacts/feature_order.json")
        self.scaler = None
        self.feature_order = None
        self.s3 = boto3.client("s3", region_name=get_env_var("AWS_REGION", "ap-south-1"))

    def load(self):
        # try local first
        scaler_local = os.path.join(self.local_dir, os.path.basename(self.scaler_key))
        feature_local = os.path.join(self.local_dir, os.path.basename(self.feature_order_key))

        if os.path.exists(scaler_local):
            try:
                self.scaler = joblib.load(scaler_local)
                logger.info("Loaded scaler from local artifacts.")
            except Exception:
                logger.exception("Failed to load local scaler; will try S3.")
                self.scaler = None

        if os.path.exists(feature_local):
            try:
                with open(feature_local, "r") as f:
                    self.feature_order = json.load(f)
                logger.info("Loaded feature_order from local artifacts.")
            except Exception:
                logger.exception("Failed to load local feature_order; will try S3.")
                self.feature_order = None

        # fallback to S3 if needed
        if (self.scaler is None or self.feature_order is None) and self.s3_bucket:
            try:
                if self.scaler is None:
                    resp = self.s3.get_object(Bucket=self.s3_bucket, Key=self.scaler_key)
                    self.scaler = joblib.load(io.BytesIO(resp["Body"].read()))
                    logger.info(f"Loaded scaler from s3://{self.s3_bucket}/{self.scaler_key}")
                if self.feature_order is None:
                    resp = self.s3.get_object(Bucket=self.s3_bucket, Key=self.feature_order_key)
                    self.feature_order = json.loads(resp["Body"].read().decode("utf-8"))
                    logger.info(f"Loaded feature_order from s3://{self.s3_bucket}/{self.feature_order_key}")
            except ClientError as e:
                logger.warning("Could not load artifacts from S3: %s", e)

        if self.feature_order is None:
            # default fallback feature order (Time, V1..V28, Amount) — numeric names
            self.feature_order = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
            logger.warning("Using default fallback feature_order (Time, V1..V28, Amount)")

    def transform_vector(self, vec: List[float]) -> List[float]:
        """
        Accepts a single transaction vector [Time, V1..V28, Amount] -> returns transformed vector.
        Only transforms Time and Amount using scaler if scaler present; otherwise returns vec unchanged.
        """
        if len(vec) != 30:
            raise ValueError("Expecting 30 features: [Time, V1..V28, Amount]")

        if self.scaler is None:
            # no scaler available, return as is
            return vec

        # Our training scaled only ["Time", "Amount"] (assumption from earlier notebook)
        # find indices:
        # Time index 0, Amount index 29
        arr = np.array(vec, dtype=float).reshape(1, -1)
        # apply scaler to first and last columns only
        # create a copy
        out = arr.copy()
        try:
            # Assuming scaler was fitted on 2 columns [Time, Amount]
            # We'll construct array with those values and transform using scaler
            small = np.array([[arr[0,0], arr[0,-1]]])
            small_t = self.scaler.transform(small)
            out[0,0] = small_t[0,0]
            out[0,-1] = small_t[0,1]
            return out.flatten().tolist()
        except Exception:
            logger.exception("Failed to apply scaler; returning original vector")
            return vec

    def transform_dataframe(self, df):
        """
        Turn a pandas DataFrame (with columns matching feature_order or raw) into scaled array.
        Returns numpy array (n_samples, 30).
        """
        # Ensure columns in the correct order
        if list(df.columns) != self.feature_order:
            try:
                df = df[self.feature_order]
            except Exception:
                # fallback: if df has extra columns, try to pick first 30 numeric columns
                df = df.iloc[:, :30]
        arr = df.values.astype(float)
        if self.scaler is None:
            return arr
        # transform cols 0 and -1
        try:
            small = arr[:, [0, -1]]
            small_t = self.scaler.transform(small)
            arr[:,0] = small_t[:,0]
            arr[:,-1] = small_t[:,1]
            return arr
        except Exception:
            logger.exception("Failed to apply scaler to dataframe; returning original array")
            return arr
