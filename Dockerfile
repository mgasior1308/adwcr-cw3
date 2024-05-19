FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app.py .
COPY model.pkl .
COPY perceptron.py .

ENV FLASK_APP=app

EXPOSE 8000
CMD ["python", "app.py"]
