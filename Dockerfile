FROM arm32v7/python:3.7-alpine

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
