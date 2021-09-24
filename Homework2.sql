-- *****************************************************************************
--                                Assignment-2
-- Host: localhost                                  Database: baseball
-- Server version: 10.6.4-MariaDB-0ubuntu0.20.04
--
-- *****************************************************************************
USE baseball;

-- **************** Historical Batting Average **********************************
-- Calculation of batting average for every batter
-- ******************************************************************************

DROP TABLE IF EXISTS batter_avg_historical;

CREATE TABLE batter_avg_historical AS 
	SELECT batter AS Batter, 
			SUM(Hit) AS Hit, 
			SUM(atBat) AS atBat,(CASE WHEN SUM(atBat) > 0 
									  THEN  SUM(Hit)/SUM(atBat) 
									  ELSE 0 
										  END) AS Batting_Avg
			FROM batter_counts 
			GROUP BY Batter;

SELECT * FROM batter_avg_historical;

-- ***************************************************************************
-- Calculation of batting average for every batter annually
-- **************** Annual Batting Average ***********************************
DROP TABLE IF EXISTS batter_avg_annual;

CREATE TABLE batter_avg_annual AS 
	SELECT batter AS Batter, 
		   YEAR(game.local_date) AS For_Year, (CASE WHEN SUM(bc.atBat) > 0 
													THEN SUM(bc.Hit)/ SUM(bc.atBat)
													ELSE 0 
														END) AS Batting_Avg
			FROM batter_counts AS bc,game
			WHERE bc.game_id = game.game_id
			GROUP BY Batter,For_Year
			ORDER BY Batter, For_Year;

SELECT * FROM batter_avg_annual;
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
SELECT * FROM batter_avg_rolling_temp bart;


CREATE OR REPLACE TABLE batter_avg_rolling AS
		SELECT bart1.game_id, bart1.batter,(CASE WHEN SUM(bart2.atBat) > 0 
											THEN SUM(bart2.Hit)/ SUM(bart2.atBat)
											ELSE 0
												END) AS Batting_Avg,
											bart1.local_date ,
											DATE_SUB(bart1.local_date, interval 100 DAY) AS since
			FROM batter_avg_rolling_temp bart1 
			INNER JOIN batter_avg_rolling_temp bart2 
			ON bart1.Batter = bart2.Batter 
			AND bart2.local_date < bart1.local_date
			AND bart2.local_date > DATE_SUB(bart1.local_date, interval 100 DAY) 
            # Remove this Where clause for all batters
			WHERE bart1.batter = 407832  
			GROUP BY bart1.Batter , bart1.local_date
			ORDER BY bart1.Batter ;


SELECT * from batter_avg_rolling;

 	