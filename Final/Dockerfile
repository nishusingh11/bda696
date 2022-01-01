FROM python:3.6
MAINTAINER Nishu Singh, nsingh2878@sdsu.edu


RUN mkdir app
WORKDIR /app
RUN mkdir ./plot

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "mariadb-client"]
RUN apt-get update && apt-get install --no-install-recommends --yes \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN /usr/local/bin/python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY ./features.sql /app/features.sql
COPY final.sh /app/final.sh
COPY ./final.py /app/final.py
COPY ./importance.py /app/importance.py
COPY ./link.py /app/link.py
COPY ./Models.py /app/Models.py
COPY msd.py /app/msd.py
COPY ./table.py /app/table.py
COPY bruteForce.py /app/bruteForce.py

RUN ["chmod","+x","/app/final.sh"]
RUN ["chmod","+x","/app/final.py"]
CMD ["/app/final.sh"]

