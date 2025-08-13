# Fraud Detection API (Enterprise Ready)

## 1. Build & Deploy
```bash
sam build --use-container
sam deploy --guided \
  --parameter-overrides EndpointName=$ENDPOINT_NAME S3Bucket=$S3_BUCKET
