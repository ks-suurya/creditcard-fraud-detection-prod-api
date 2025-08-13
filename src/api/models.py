from pydantic import BaseModel, Field, conlist
from typing import List, Optional

# Single transaction: ordered numeric features as a flat list
# Example: [Time, V1, V2, ..., V28, Amount] = 30 values
TransactionVector = conlist(float, min_length=30, max_length=30)

class PredictRequest(BaseModel):
    features: TransactionVector = Field(
        ..., description="Ordered list: [Time, V1..V28, Amount]"
    )

class PredictResponse(BaseModel):
    probability: float
    label: int

class PredictBatchRequest(BaseModel):
    transactions: List[TransactionVector]

class PredictBatchResponse(BaseModel):
    probabilities: List[float]
    labels: List[int]

class HealthResponse(BaseModel):
    status: str
    endpoint: str
