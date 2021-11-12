# BA Hit/Atbat
#TB – Total bases: one for each single, two for each double, three for each triple, and four for each home run 
#                  [1B + (2 × 2B) + (3 × 3B) + (4 × HR)]
#SLG – Slugging average: total bases achieved on hits divided by at-bats (TB/AB)
#OBP – On-base percentage: times reached base (H + BB + HBP) divided by at bats plus walks plus hit by pitch plus sacrifice flies (AB + BB + Hit_by_pitch + SF)
# XBH= 2B+3B+HR
# TOB – Times on base: times reaching base as a result of hits, walks, and hit-by-pitches (Hit + walk + Hit_by_pitch)
#OPS
#ERA
#IP
#WHIP



Use newbaseball;

DROP TABLE IF EXISTS rolling_100_days;
CREATE TABLE IF NOT EXISTS rolling_100_days AS
SELECT tbc1.team_id, tbc1.game_id, COUNT(*) AS count,
       SUM(tbc2.atBat) AS atBat,
       SUM(tbc2.Hit) AS Hit,
       COUNT(tbc2.Hit) As Total_hit,
       SUM(tbc2.toBase) AS Base,
       SUM(tbc2.`Double`) AS `Double`,
       SUM(tbc2.Double_Play) AS Double_Play,
       SUM(tbc2.Fly_Out) +  SUM(tbc2.Flyout) AS Fly_Out,
       SUM(tbc2.Hit_By_Pitch) AS Hit_By_Pitch,
       SUM(tbc2.Home_Run) AS Home_Run,
       SUM(tbc2.Sac_Fly) AS Sac_Fly,
       SUM(tbc2.Single) AS Single,
       SUM(tbc2.Triple) AS Triple,
       SUM(tbc2.Triple_Play) AS Triple_Play,
       SUM(tbc2.Walk) AS Walk,
       #AVG(b.away_runs) AS Away_Runs,
       #AVG((b.away_runs)/((tps.endingInning-tps.startingInning)+0.0001)) AS for_ERA,
       nullif(9*(avg(b.away_runs)/((tps.endingInning-tps.startingInning)+0.0001)),0) AS ERA,
       SUM(tps.endingInning-tps.startingInning) AS IP,
	 #nullif(sum(tps.endingInning-tps.startingInning),0) AS IP,
       SUM(tps.walk) AS P_walk,
       SUM(tps.Hit) AS P_Hit
		#((SUM(tps.walk)+SUM(tps.Hit))/(SUM(tps.endingInning-tps.startingInning)+0.0001)) AS WHIP 
FROM team_batting_counts tbc1
JOIN game g1 ON   g1.game_id = tbc1.game_id 
JOIN team_pitching_stat tps ON g1.game_id =tps.game_id 
JOIN boxscore b ON g1.game_id =b.game_id
JOIN team_batting_counts tbc2
ON   tbc1.team_id = tbc2.team_id 
JOIN game  g2
ON g2.game_id = tbc2.game_id
AND g2.local_date < g1.local_date 
AND g2.local_date >= DATE_ADD(g1.local_date, INTERVAL -100 DAY)
GROUP BY tbc1.team_id, tbc1.game_id
ORDER BY tbc1.team_id ;

CREATE UNIQUE INDEX team_game ON rolling_100_days(team_id, game_id);
SELECT * FROM rolling_100_days;
SELECT COUNT(*)  FROM rolling_100_days;


# Calculating features 
DROP TABLE IF EXISTS features;
CREATE TABLE IF NOT EXISTS features AS
SELECT g.game_id, 
       g.home_team_id AS home_team_id,
       g.away_team_id AS away_team_id,
       r2dh.Hit/r2dh.atBat AS BA_home,
       r2da.Hit/r2da.atBat AS BA_away,
       r2dh.atBat/(r2dh.Home_Run+0.0001) AS A_to_HR_home,
       r2da.atBat/ (r2da.Home_Run+0.0001) AS A_to_HR_away,
       r2dh.Home_Run/(r2dh.Hit+0.0001)  AS HR_per_hit_home,
       r2da.Home_Run/(r2da.Hit+0.0001) AS HR_per_hit_away,
       r2dh.Hit + r2dh.`Double`*2 + r2dh.Triple*3 + r2dh.Home_Run*4 AS TB_home,
       r2da.Hit + r2da.`Double`*2 + r2da.Triple*3 + r2da.Home_Run*4 AS TB_away,
       r2dh.Hit + r2dh.Walk + r2dh.Hit_By_Pitch AS TOB_home,
       r2da.Hit + r2da.Walk + r2da.Hit_By_Pitch AS TOB_away,
       r2dh.`Double` + r2dh.Triple + r2dh.Home_Run AS XBH_home,
       r2da.`Double` + r2da.Triple + r2da.Home_Run AS XBH_away,
       (r2dh.Hit + r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat + r2dh.Walk + r2dh.Hit_By_Pitch + r2dh.Sac_Fly) AS OBP_home,
       (r2da.Hit + r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat + r2da.Walk + r2da.Hit_By_Pitch + r2da.Sac_Fly) AS OBP_away,
       (r2dh.Single + r2dh.`Double`*2 + r2dh.Triple*3 + r2dh.Home_Run*4)/(r2dh.atBat) AS SLG_home,
       (r2da.Single + r2da.`Double`*2 + r2da.Triple*3 + r2da.Home_Run*4)/(r2da.atBat) AS SLG_away,
       (((r2dh.Single + r2dh.`Double`*2 + r2dh.Triple*3 + r2dh.Home_Run*4)/(r2dh.atBat))-(r2dh.Hit/r2dh.atBat)) AS ISO_home,
       (((r2da.Single + r2da.`Double`*2 + r2da.Triple*3 + r2da.Home_Run*4)/(r2da.atBat))-(r2da.Hit/r2da.atBat)) AS ISO_away,
       (((r2dh.Hit + r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat + r2dh.Walk + r2dh.Hit_By_Pitch + r2dh.Sac_Fly))+
       ((r2dh.Single + r2dh.`Double`*2 + r2dh.Triple*3 + r2dh.Home_Run*4)/(r2dh.atBat))) AS OPS_home,
       (((r2da.Hit + r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat + r2da.Walk + r2da.Hit_By_Pitch + r2da.Sac_Fly))+
       ((r2da.Single + r2da.`Double`*2 + r2da.Triple*3 + r2da.Home_Run*4)/(r2da.atBat))) AS OPS_away,
       #(9*r2dh.for_ERA) AS ERA_home,
       #(9*r2da.for_ERA) AS ERA_away,
       r2dh.ERA AS ERA_home,
       r2da.ERA AS ERA_away,
       r2dh.IP AS IP_home,
       r2da.IP AS IP_away,
       #((SUM(tps.walk)+SUM(tps.Hit))/(SUM(tps.endingInning-tps.startingInning)+0.0001)) AS WHIP 
       ((r2dh.P_walk+r2dh.P_Hit)/(r2dh.IP+0.0001)) AS WHIP_home,
       ((r2da.P_walk+r2da.P_Hit)/(r2da.IP+0.0001)) AS WHIP_away
FROM game g
JOIN rolling_100_days r2dh ON g.game_id = r2dh.game_id AND g.home_team_id = r2dh.team_id
JOIN rolling_100_days r2da ON g.game_id = r2da.game_id AND g.away_team_id = r2da.team_id; 
SELECT * FROM features;
SELECT COUNT(*) FROM features;




# creating game_results table

DROP TABLE IF EXISTS response;
CREATE TABLE IF NOT EXISTS response AS
SELECT game_id AS g_id, 
       CASE 
          WHEN winner_home_or_away = 'H' THEN 1
          WHEN winner_home_or_away = 'A' THEN 0
          ELSE 0 END AS home_team_wins
FROM boxscore;

# creating intermediary predictor table

DROP TABLE IF EXISTS feature_with_response;
CREATE TABLE IF NOT EXISTS feature_with_response AS
SELECT * 
#FROM baseball_features_3 bs3
FROM features f 
JOIN response r 
ON   f.game_id = r.g_id
ORDER BY f.game_id;
SELECT * FROM feature_with_response;

# dropping redundant columns

ALTER TABLE feature_with_response 
DROP COLUMN g_id;


# creating final features table

DROP TABLE IF EXISTS features_ratio;
CREATE TABLE IF NOT EXISTS features_ratio AS
SELECT 
     game_id, home_team_id, away_team_id,
     ROUND(BA_home/(BA_away+0.0001),3)  AS BA_ratio,
     ROUND(A_to_HR_home/(A_to_HR_away+0.0001),3) AS A_to_HR_ratio,
     ROUND(HR_per_hit_home/(HR_per_hit_away+0.0001),3) AS HR_per_hit_ratio,
     ROUND(TB_home/(TB_away+0.0001),3) AS TB_ratio,
     ROUND(TOB_home/(TOB_away+0.0001),3) AS TOB_ratio,
     ROUND(XBH_home/(XBH_away+0.0001),3) AS XBH_ratio,
     ROUND(OBP_home/(OBP_away+0.0001),3) AS OBP_ratio,
     ROUND(SLG_home/(SLG_away+0.0001),3) AS SLG_ratio,
     ROUND(OPS_home/(OPS_away+0.0001),3) AS OPS_ratio,
     ROUND(ISO_home/(ISO_away+0.0001),3) AS ISO_ratio,
     ROUND((ERA_home/(ERA_away+0.0001)),3) AS ERA_ratio,
     ROUND((IP_home/(IP_away+0.0001)),3) AS IP_ratio,
     ROUND((WHIP_home/(WHIP_away+0.0001)),3) AS WHIP_ratio,
     home_team_wins
FROM feature_with_response
ORDER BY game_id;

SELECT * FROM features_ratio;
SELECT COUNT(*) FROM features_ratio;


DROP TABLE IF EXISTS features_Diff;
CREATE TABLE IF NOT EXISTS features_Diff AS
SELECT 
     game_id, home_team_id, away_team_id,
     ROUND(((BA_home) -(BA_away)),3) AS BA_Diff,
     ROUND(((A_to_HR_home) - (A_to_HR_away)),3)  AS A_to_HR_Diff,
     ROUND((HR_per_hit_home - HR_per_hit_away),3) As HR_per_hit_Diff,
     ROUND((TB_home - TB_away),3) AS TB_Diff,
     ROUND((TOB_home - TOB_away),3) AS TOB_Diff,
     ROUND((XBH_home - XBH_away),3) AS XBH_Diff,
     ROUND((OBP_home - OBP_away),3) AS OBP_Diff,
     ROUND((SLG_home - SLG_away),3) AS SLG_Diff,
     ROUND((OPS_home - OPS_away),3) AS OPS_Diff,
     ROUND((ISO_home - ISO_away),3) AS ISO_Diff,
     ROUND((ERA_home - ERA_away),3) AS ERA_Diff,
     ROUND(((IP_home)-(IP_away)),3) AS IP_Diff,
     ROUND(((WHIP_home)-(WHIP_away)),3) AS WHIP_Diff,
     home_team_wins
FROM feature_with_response
ORDER BY game_id;


SELECT * FROM features_Diff;
SELECT COUNT(*) FROM features_Diff;


DROP TABLE IF EXISTS features_per;
CREATE TABLE IF NOT EXISTS features_per AS
SELECT 
     game_id, home_team_id, away_team_id,
     ROUND((((BA_home -BA_away)/(BA_home+0.0001))*100),3) AS BA_Diff_Per,
     ROUND((((A_to_HR_home - A_to_HR_away)/(A_to_HR_home+0.0001))*100),3) AS A_to_HR_Diff_Per,
     ROUND(((HR_per_hit_home - HR_per_hit_away)/(HR_per_hit_home+0.0001)*100),3) AS HR_per_hit_Diff_Per,
     ROUND((((TB_home - TB_away)/(TB_home+0.0001))*100),3) AS TB_Diff_Per,
     ROUND((((TOB_home - TOB_away)/(TOB_home+0.0001))*100),3) AS TOB_Diff_Per,
     ROUND((((XBH_home - XBH_away)/(XBH_home+0.0001))*100),3) AS XBH_Diff_Per,
     ROUND((((OBP_home - OBP_away)/(OBP_home+0.0001))*100),3) AS OBP_Diff_Per,
     ROUND((((SLG_home - SLG_away)/(SLG_home+0.0001))*100),3) AS SLG_Diff_Per,
     ROUND((((OPS_home - OPS_away)/(OPS_home+0.0001))*100),3) AS OPS_Diff_Per,
     ROUND((((ISO_home - ISO_away)/(ISO_home+0.0001))*100),3) AS ISO_Diff_Per,
     ROUND((((IP_home - IP_away)/(IP_home+0.0001))*100),3) AS IP_Diff_Per,
     home_team_wins
FROM feature_with_response
ORDER BY game_id;


SELECT * FROM features_per;
SELECT COUNT(*) FROM features_per;

#SELECT * from batter_counts;
#SELECT * from boxscore;




