FROM python:3.9

WORKDIR /image_classifier

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

ENV PYTHONPATH /image_classifier

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]