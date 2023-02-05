FROM python:3.10-slim AS builder
ENV WORKDIR /app
WORKDIR $WORKDIR
ADD . $WORKDIR
RUN apt update && apt install build-essential python-six cmake git -y
RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir -r requirements.txt 
RUN apt install libsndfile1 -y
