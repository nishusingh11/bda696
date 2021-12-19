#!/bin/sh



echo "sleeping for 10 sec"
sleep 10
echo "checking and loading baseball database"

if ! mysql -h bda696_mariadb -u root -pabc123 -e 'use baseball;'; then

  echo "baseball database does not exists"
  mysql -h bda696_mariadb -u root -pabc123 -e "create database baseball;"
  echo "loading database..."
  mysql -u root -pabc123 -h bda696_mariadb --database=baseball < /app/res/baseball.sql

  else
    echo "baseball database exists"
  fi

mysql -h bda696_mariadb -u root -pabc123 baseball < /app/features.sql
mysql -h bda696_mariadb -u root -pabc123 baseball -e '
 SELECT * FROM feature_per; '> /app/plot/result.txt
 echo "features data file stored in result.txt"

 # python script

 python final.py
 python importance.py
 python link.py
 python msd.py
 python Models.py
 python bruteForce.py
 python table.py
 echo "python script execution done"