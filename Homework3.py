import sys

from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SparkSession
from batting_avg_transform import BattingAverageTransform


def main():

    # Building spark session
    conf = SparkConf().set("spark.jars", "./jars/mariadb-java-client-2.7.4.jar")
    SparkContext(conf=conf)
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.sql.debug.maxToStringFields", 128)
        .getOrCreate()
    )
    user = "root"  # pragma: allowlist secret
    password = "abc123"  # pragma: allowlist secret

    # Dataframe for batter_counts table
    batter_dataframe = (
        spark.read.format("jdbc")
        .options(
            url="jdbc:mysql://localhost:3306/newbaseball",
            driver="org.mariadb.jdbc.Driver",
            dbtable="batter_counts",
            user=user,
            password=password,
        )
        .load()
    )
    batter_dataframe.createOrReplaceTempView("batter_counts")
    batter_dataframe.persist(StorageLevel.MEMORY_AND_DISK)

    # Dataframe for game table
    game_dataframe = (
        spark.read.format("jdbc")
        .options(
            url="jdbc:mysql://localhost:3306/newbaseball",
            driver="org.mariadb.jdbc.Driver",
            dbtable="game",
            user=user,
            password=password,
        )
        .load()
    )
    game_dataframe.createOrReplaceTempView("game")
    game_dataframe.persist(StorageLevel.MEMORY_AND_DISK)

    # Dataframe rolling lookup
    rolling_lookup_df = (
        spark.read.format("jdbc")
        .options(
            url="jdbc:mysql://localhost:3306/newbaseball",
            driver="org.mariadb.jdbc.Driver",
            dbtable="(SELECT g.game_id , local_date , batter, atBat , Hit \
                                FROM batter_counts as bc JOIN game as g ON g.game_id = bc.game_id \
                                WHERE atBat > 0 \
                                ORDER BY batter, local_date) batter_avg_temp",
            user=user,
            password=password,
        )
        .load()
    )
    # rolling_lookup_df.show()
    rolling_lookup_df.createOrReplaceTempView("batter_avg_temp")
    rolling_lookup_df.persist(StorageLevel.MEMORY_AND_DISK)

    # Sql query to calculate the total hits and total atBat.
    rolling_sum = spark.sql(
        """ SELECT
                bart1.batter
                , bart1.game_id
                , bart1.local_date
                , SUM(bart2.Hit) AS Total_Hit
                , SUM(bart2.atBat) As Total_atBat
                FROM batter_avg_temp bart1 JOIN
                batter_avg_temp bart2 ON
                bart1.batter = bart2.batter AND
                bart2.local_date BETWEEN DATE_SUB(bart1.local_date, 100) AND
                bart1.local_date
                GROUP BY
                bart1.batter, bart1.game_id, bart1.local_date
                """
    )

    rolling_lookup_df.createOrReplaceTempView("batter_avg_temp")
    rolling_lookup_df.persist(StorageLevel.MEMORY_AND_DISK)

    # Using transformation for calculation of rolling average.
    batting_avg_transform = BattingAverageTransform(
        inputCols=["Total_Hit", "Total_atBat"], outputCol="Batting_Average"
    )
    rolling_average = batting_avg_transform.transform(rolling_sum)
    print("+----------------Batting Average for Rolling over 100 Days---------------+")

    # Printing final Rolling average table
    #rolling_average.show(n=100, truncate=False)
    #rolling_average.write.csv('output')
    rolling_average.show()

if __name__ == "__main__":
    sys.exit(main())
