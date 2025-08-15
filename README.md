# Fraud Detection API (Enterprise Ready)
## 1st Docker: docker run --env-file .env -p 8000:8000 -v "${env:USERPROFILE}\.aws:/root/.aws:ro" -v "${PWD}\logs:/app/logs" fraud-api:latest
## 1. Build & Deploy
```bash
sam build --use-container
sam deploy --guided \
  --parameter-overrides EndpointName=$ENDPOINT_NAME S3Bucket=$S3_BUCKET


