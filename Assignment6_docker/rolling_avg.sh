#!/bin/sh

# 1. for sql shell scripting, took reference from stackoverflow
#    https://stackoverflow.com/questions/7616520/how-to-execute-a-sql-script-from-bash
#2. using sleep, to start the bda696_mariadb service completely.
#   Otherwise you might get error "can not connect mysql_service".
#   Ref https://stackoverflow.com/questions/31746182/docker-compose-wait-for-container-x-before-starting-y
#3. Reusing homework2.sql with little changes.


sleep 5
echo "checking and loading baseball database"
if ! mysql -h bda696_mariadb -u root -pabc123 -e 'USE baseball'; then

  echo "baseball database does not exists"
  mysql -h bda696_mariadb -u root -pabc123 -e create baseball
  echo "loading database..."
  mysql -u root -pabc123 -h bda696_mariadb --database=baseball < /app/sql/baseball.sql

  else
    echo "baseball database exists"
  fi

mysql -h bda696_mariadb -u root -pabc123 baseball < /app/sql/Homework2.sql
mysql -h bda696_mariadb -u root -pabc123 baseball -e '
 SELECT * FROM batter_avg_rolling; '> /app/output/rolling_table.txt
 echo "result data stored in result folder"