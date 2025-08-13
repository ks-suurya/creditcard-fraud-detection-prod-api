FROM public.ecr.aws/lambda/python:3.10

# Copy source code
COPY src/ ${LAMBDA_TASK_ROOT}
COPY requirements.txt  .

# Install dependencies
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN mkdir -p /app/logs

CMD ["handler.lambda_handler"]
