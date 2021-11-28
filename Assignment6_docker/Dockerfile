FROM ubuntu:latest
MAINTAINER Nishu Singh, nsingh2878@sdsu.edu

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN mkdir ./sql
RUN mkdir ./script
RUN mkdir ./output

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "mariadb-client"]

COPY ./baseball.sql ./sql
COPY ./Homework2.sql ./sql
COPY rolling_avg.sh ./script
RUN ["chmod","+x","./script/rolling_avg.sh"]
ENTRYPOINT ["/usr/bin/sh","-c"]
CMD ["./script/rolling_avg.sh"]

