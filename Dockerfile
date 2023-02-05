FROM python:3.10-slim AS builder
ENV WORKDIR /app
WORKDIR $WORKDIR
ADD . $WORKDIR
RUN apt update && apt install build-essential python-six git -y
RUN pip install --upgrade --no-cache-dir pip && pip install six && pip install --no-cache-dir -r requirements.txt && apt install libsndfile1 -y
