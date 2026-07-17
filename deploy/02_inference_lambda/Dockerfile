# Use AWS base image for Python 3.10
FROM public.ecr.aws/lambda/python:3.10

# IMPORTANT: Install CPU-only PyTorch. This drops the image size from ~2.5GB to ~700MB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install boto3
RUN pip install --no-cache-dir boto3

# Copy code
COPY lambda_inference.py ${LAMBDA_TASK_ROOT}
COPY model.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_inference.lambda_handler" ]