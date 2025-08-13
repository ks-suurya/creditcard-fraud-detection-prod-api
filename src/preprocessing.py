import numpy as np

def preprocess(instances, scaler, feature_order):
    processed = []
    for inst in instances:
        row = [inst.get(feat, 0) for feat in feature_order]
        processed.append(row)
    arr = np.array(processed)
    arr_scaled = scaler.transform(arr)
    return arr_scaled
