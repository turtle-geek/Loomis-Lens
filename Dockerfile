FROM public.ecr.aws/lambda/python:3.11

# Install ONLY the runtime libraries for OpenCV (no compilers)
RUN yum update -y && yum install -y \
    mesa-libGL \
    glib2 \
    && yum clean all

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies - FORCE BINARY ONLY
# This prevents it from ever trying to compile NumPy or h5py
RUN pip install --default-timeout=1000 --no-cache-dir \
    --only-binary=:all: \
    -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy all application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "app.handler" ]