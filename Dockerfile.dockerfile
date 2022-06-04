
FROM python:3.6-slim

WORKDIR /app/


COPY requirements.txt /app/

RUN pip install -r ./requirements.txt

COPY app.py __init__.py /app/

COPY model.pkl /app/
