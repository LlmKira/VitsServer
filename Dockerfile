# Stage 1 - Build dependencies from source
FROM python:3.8-slim AS builder

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y build-essential libsndfile1 vim gcc g++ cmake gfortran libopenblas-dev liblapack-dev cython && \
    python3 -m pip install --upgrade pip numpy numba scipy

WORKDIR /build

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

# Stage 2 - Runtime image
FROM python:3.8-slim

ENV SERVER_HOST='0.0.0.0' \
    SERVER_PORT=9557

EXPOSE $SERVER_PORT

WORKDIR /app

COPY --from=builder /usr/local/ /usr/local/

COPY . .

CMD ["python3", "main.py"]
