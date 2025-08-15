# Real-Time Credit Card Fraud Detection API on AWS

This project demonstrates a production-ready, cloud-native solution for detecting fraudulent credit card transactions in real-time. It leverages a machine learning model deployed as a secure, high-performance API on Amazon Web Services (AWS).

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture & Workflows](#system-architecture--workflows)
3. [Technical Stack](#technical-stack)
4. [Deployment Instructions](#deployment-instructions)
    - [Local Docker Deployment](#local-docker-deployment)
    - [AWS Cloud Deployment](#aws-cloud-deployment)
5. [Key Implementations](#key-implementations)
6. [Future Work](#future-work)

## Project Overview

The core of this project is a RESTful API that provides real-time fraud predictions for credit card transactions. The system is built to be scalable, secure, and resilient, following modern MLOps and cloud architecture best practices. An XGBoost model, trained on a transaction dataset, serves as the prediction engine. The entire application is containerized with Docker and orchestrated on the cloud, showcasing an end-to-end deployment workflow.

## System Architecture & Workflows

The infrastructure is designed for scalability and high availability on AWS. The system supports two distinct prediction workflows: real-time single transaction scoring and asynchronous batch processing.

1.  **Amazon ECR (Elastic Container Registry)**: Stores the versioned Docker images of the FastAPI application.
2.  **Amazon ECS (Elastic Container Service)**: Orchestrates the deployment of the Docker containers. It runs the application on **AWS Fargate**, a serverless compute engine, which automatically manages the underlying infrastructure.
3.  **Amazon SageMaker**: Hosts the trained XGBoost model as a real-time inference endpoint. The ECS task communicates with this endpoint to get predictions.
4.  **IAM (Identity and Access Management)**: Provides granular, secure access control. An `ecsTaskExecutionRole` is used to grant the container permissions to pull images and write logs, while a `taskRoleArn` grants the application itself permission to invoke the SageMaker endpoint.
5.  **Amazon CloudWatch**: Centralizes logging for the running application, capturing output and errors for monitoring and debugging.
6.  **VPC & Networking**: The service is deployed within a custom Virtual Private Cloud (VPC) with public subnets and a security group configured to allow inbound traffic only on port 8000.

### Prediction Workflows

* **Real-Time Workflow (`/predict`)**: A user sends a single transaction to the API. The FastAPI application immediately preprocesses the data and invokes the SageMaker endpoint. A fraud score is returned synchronously in the API response. This is designed for immediate, low-latency decisions.
* **Batch Workflow (`/predict/batch`)**: A user sends a list of multiple transactions. The API processes them efficiently and returns a list of fraud scores. This is ideal for scoring multiple transactions without the overhead of making individual API calls.

## Technical Stack
- **Programming Language**: Python 3.10
- **API Framework**: FastAPI with Uvicorn
- **Machine Learning**: Scikit-learn, XGBoost
- **Containerization**: Docker, Docker Compose
- **Cloud Provider**: Amazon Web Services (AWS)
- **Infrastructure as Code**: AWS CLI, PowerShell
- **CI/CD**: (Placeholder for future implementation, e.g., GitHub Actions)

## Deployment Instructions

You can run this application both locally using Docker or deploy it to your own AWS account.

### Local Docker Deployment

This method is perfect for testing and development.

**Prerequisites:**
* Docker and Docker Compose installed.
* Python 3.10.

**Steps:**
1.  **Clone the Repository:**
    ```
    git clone https://github.com/ks-suurya/creditcard-fraud-detection-prod-api.git
    cd creditcard-fraud-detection-prod-api
    ```
2.  **Create an Environment File:** Create a file named `.env` in the root of the project and add the required environment variables.
    ```
    # A secure, random key for your API
    API_KEY=<Generate a random key>
    
    # The name of your SageMaker endpoint (can be a placeholder for local testing)
    SAGEMAKER_ENDPOINT=your-sagemaker-endpoint-name
    
    # Your AWS Region
    AWS_REGION=your-aws-region
    ```
3.  **Build and Run the Container:** Use the following command to build the Docker image and start the service.
    ```
    docker build -f Dockerfile.fastapi -t fraud-api:latest .
    docker run --env-file .env -p 8000:8000 fraud-api:latest
    ```
4.  **Test the API:** The API will be available at `http://localhost:8000`. You can now send requests to it using a tool like Postman or `curl`.

### AWS Cloud Deployment

This section provides a high-level guide to deploying the service on AWS ECS with Fargate.

**Prerequisites:**
* An AWS account.
* AWS CLI installed and configured with your credentials.
* Docker installed.
* A trained SageMaker endpoint deployed and running.

**Steps:**
1.  **Build and Push the Docker Image to ECR:**
    * Create an ECR repository to store your image.
    * Build your Docker image and tag it with the ECR repository URI.
    * Authenticate Docker with ECR and push the image.
2.  **Set Up IAM Roles:**
    * Create an **ECS Task Execution Role** (`ecsTaskExecutionRole`) that grants permissions for ECS to pull images from ECR and write logs to CloudWatch.
    * Create a **Task Role** with a policy that allows the `sagemaker:InvokeEndpoint` action on your specific SageMaker endpoint ARN. This role will be assigned to the `taskRoleArn` in the task definition.
3.  **Configure the Task Definition:**
    * Open the `task-def.json` file.
    * Update the `image` URI to point to the image you pushed to ECR (e.g., `your-account-id.dkr.ecr.your-region.amazonaws.com/fraud-api:latest`).
    * Update the `executionRoleArn` and `taskRoleArn` with the ARNs of the roles you created.
    * Update the `SAGEMAKER_ENDPOINT` environment variable with the name of your live SageMaker endpoint.
4.  **Deploy the Service:**
    * Create an ECS Cluster.
    * Register the task definition using the AWS CLI: `aws ecs register-task-definition --cli-input-json file://task-def.json`.
    * Create a Security Group that allows inbound traffic on port `8000`.
    * Create the ECS Service, specifying your cluster, task definition, subnets, and security group. Ensure you enable `assignPublicIp` to make the API accessible.

## Key Implementations

### Machine Learning Pipeline

* **Preprocessing**: A `Preprocessor` applies scaling to normalize transaction data. Artifacts (e.g., scaler and feature order) ensure the exact same transformations are applied during inference as in training.
* **Model Integration**: Requests are formatted for a SageMaker XGBoost endpoint (e.g., `text/csv` payloads).

### API Development

* **High-Performance Endpoints**: Asynchronous endpoints for single (`/predict`) and batch (`/predict/batch`) predictions.
* **Data Validation**: Pydantic models enforce request body validation with clear error messages for invalid inputs.

### Containerization & CI/CD

* **Dockerfile**: Multi-stage `Dockerfile.fastapi` for a lean, production-optimized container image.
* **Local Environment**: `docker-compose.yml` and a `.env` file help replicate production variables locally.

### Cloud Infrastructure

* **Infrastructure Scripting**: AWS CLI and PowerShell scripts automate IAM roles, ECR repositories, security groups, and ECS service creation.
* **Serverless Compute**: Deployed on **AWS Fargate**, removing the need to manage EC2 instances.

### Security

* **API Key Authentication**: API key protection for prediction endpoints.
* **IAM Least Privilege**: Separate Execution and Task roles for infrastructure-level and application-level permissions.

## Future Work

* **Monitoring & Alerting**: Set up CloudWatch Alarms to monitor API latency, error rates, and model performance, with alerts sent via SNS.
* **Load Balancing**: Introduce an Application Load Balancer (ALB) in front of the ECS service to enable auto-scaling and improve availability.