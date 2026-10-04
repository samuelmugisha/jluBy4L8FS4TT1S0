# syntax=docker/dockerfile:1
FROM python:3.11-bookworm

LABEL maintainer="samuelmugisha <dcgroup205@gmail.com>"

RUN mkdir -p /vcfad-webapp

WORKDIR /vcfad-webapp

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "5000"]
