-- *****************************************************************************
--                                Assignment-2
-- Host: localhost                                  Database: baseball
-- Server version: 10.6.4-MariaDB-0ubuntu0.20.04
--
-- *****************************************************************************
USE baseball;

-- ****************************************************************************
-- Calculation of batting average for rolling over last 100 days,
-- created temporary table for storing intermediate results
-- **************** Rolling Batting Average ***********************************

CREATE OR REPLACE TEMPORARY TABLE batter_avg_rolling_temp AS
 	   SELECT   game.game_id,
 	         bc.batter AS Batter,
 			 bc.Hit, bc.atBat,
 			 game.local_date
			 FROM batter_counts AS bc,game
			 WHERE bc.game_id = game.game_id;
#CREATE INDEX Batter_id ON batter_avg_rolling_temp(Batter);
#SELECT * FROM batter_avg_rolling_temp bart;


CREATE OR REPLACE TABLE batter_avg_rolling AS
		SELECT bart1.batter,(CASE WHEN SUM(bart2.atBat) > 0
											THEN SUM(bart2.Hit)/ SUM(bart2.atBat)
											ELSE 0
											END) AS Batting_Avg,
											bart1.game_id,
											bart1.local_date ,
											DATE_SUB(bart1.local_date, interval 100 DAY) AS since
			FROM batter_avg_rolling_temp bart1
			INNER JOIN batter_avg_rolling_temp bart2
			ON bart1.Batter = bart2.Batter
			AND bart2.local_date < bart1.local_date
			AND bart2.local_date > DATE_SUB(bart1.local_date, interval 100 DAY)
			# Remove this Where clause for all players
			WHERE bart1.game_id = 12560
			GROUP BY bart1.Batter , bart1.local_date
			ORDER BY bart1.Batter;



#SELECT * from batter_avg_rolling;
#SELECT count(*) FROM batter_avg_rolling;

 	