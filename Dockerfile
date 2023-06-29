FROM python:3.9
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         ffmpeg libsm6 libxext6 nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV MODEL_BASE_PATH=/opt/ml/models

WORKDIR /opt/program
RUN mkdir models
COPY requirements.txt /opt/program/inference/requirements.txt
RUN pip install -r inference/requirements.txt

# Set up the program in the image
COPY . .

RUN chmod +x ./serve

CMD ./serve