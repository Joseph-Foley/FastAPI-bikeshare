FROM tiangolo/uvicorn-gunicorn:python3.8
COPY ./app /app
WORKDIR /app
RUN pip install -r requirements.txt
