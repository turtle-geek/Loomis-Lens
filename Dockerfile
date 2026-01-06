# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Install system libraries required for OpenCV and MediaPipe
RUN yum update -y && yum install -y \
    mesa-libGL \
    glib2 \
    && yum clean all

# Copy requirements.txt directly to the Lambda task root
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies into the task root
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy all application code to the task root
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to point to your Mangum handler in app.py
CMD [ "app.handler" ]