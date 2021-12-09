# Features Seelcted:
# Batting Statistics
# 1. BA =  Hit/Atbat
# 2. A_HR = atBat/Home_RUN
# 3. HBP = Hit by Pitch
# 4. HRH = Home Run per Hit
# 5. ISO = Isolated power (TB-H/atBAt)
# 6. SLG = Slugging Average (TB/atBAt)
# 7. OBP = on base percentage (H+BB+HBP)/(atBAt+BB+HBP+Sac_Fly)
# 8. TB = Total Bases [H + 2B + (2 × 3B) + (3 × HR)] or [1B + (2 × 2B) + (3 × 3B) + (4 × HR)]
# 17. XBH = Extra bases hits XBH = 2B + 3B + HR
# 9. TOB = Times on base (H + BB + HBP)
# Pitching Statistics
# 10. IP = Inning Pitched (endingInning-startingInning)
# 11. ERA = earned run average 9*(earned run allowed/IP)
# 12. WHIP = walks plus hits per inning pitched
# 13. FIP=
#14. P_BB/9= bases on balls per nine innins pitched, 9*(BB/IP)
#15 P_K9 =
#16 HR9
# 18. DICE = 3.00+((13*HR)+(3*(BB+HBP))-2K)/IP
# 19: KBB= strikeout to walk ratio: K/BB






USE baseball;

DROP TEMPORARY TABLE IF EXISTS team_pitch_temp;
CREATE TEMPORARY TABLE team_pitch_temp ENGINE=MEMORY AS
SELECT tpc.game_id,
tpc.team_id,
tpc.win AS P_win,
tpc.atBat AS P_atBat,
tpc.Hit As P_Hit,
tpc.finalScore As P_finalScore,
tpc.plateApperance As P_plateApperance,
tpc.Fly_Out As P_Fly_Out,
tpc.Hit_By_Pitch As P_Hit_By_Pitch,
tpc.Home_Run As P_Home_Run,
tpc.Strikeout AS P_Strikeout,
tpc.Walk As P_Walk,
pc.startingPitcher,
SUM(pc.startingInning) AS P_startingInning ,
SUM(pc.endingInning) AS P_endingInning
FROM team_pitching_counts tpc
JOIN pitcher_counts pc ON pc.game_id = tpc.game_id
AND pc.team_id = tpc.team_id
GROUP BY pc.team_id,pc.game_id;
CREATE INDEX team_pitch_temp_idx ON team_pitch_temp (team_id, game_id);

SELECT * FROM team_pitch_temp;



DROP TABLE IF EXISTS feature_temp;
CREATE TABLE IF NOT EXISTS feature_temp ENGINE=MEMORY AS
SELECT tbc.*,
b.away_runs,
tr.home_streak,
tr.away_streak,
tpt.P_win,
tpt.P_atBat,
tpt.P_Hit,
tpt.P_Hit_By_Pitch,
tpt.P_Home_Run,
tpt.P_Strikeout,
tpt.P_Walk,
(tpt.P_startingInning) AS P_startingInning,
(tpt.P_endingInning) AS P_endingInning
FROM team_batting_counts tbc
JOIN team_pitch_temp tpt ON tbc.game_id = tpt.game_id AND tbc.team_id = tpt.team_id
JOIN team_results tr ON tr.team_id = tbc.team_id AND tr.game_id = tbc.game_id
JOIN boxscore b ON tbc.game_id = b.game_id GROUP BY team_id, game_id;

SELECT * FROM feature_temp;


DROP TEMPORARY TABLE IF EXISTS rolling_100_days;
CREATE TEMPORARY TABLE  rolling_100_days ENGINE=MEMORY AS
SELECT
ft1.team_id as team_id,
ft1.game_id as game_id,
NULLIF (SUM(ft2.atBat),0) AS atBat,
NULLIF (SUM(ft2.Hit),0) AS Hit,
NULLIF(SUM(ft2.Single),0) AS B,
NULLIF(SUM(ft2.Double),0) AS 2B,
NULLIF(SUM(ft2.Triple),0) AS 3B,
NULLIF(SUM(ft2.Home_Run),0) AS Home_Run,
NULLIF(SUM(ft2.Sac_Fly),0) AS Sac_Fly,
NULLIF(SUM(ft2.Walk),0) AS BB,
NULLIF(SUM(ft2.Fly_Out),0) AS Fly_Out,
NULLIF(SUM(ft2.Hit_By_Pitch),0) AS Hit_By_Pitch,
NULLIF(SUM(ft2.Single+(2*ft2.Double)+(3*ft2.Triple)+(4*ft2.Home_Run)),0) AS TB,
NULLIF(SUM(ft2.Hit+ft2.Walk+ft2.Hit_By_Pitch),0) AS TOB,
NULLIF(SUM(ft2.P_Walk),0) AS P_BB,
NULLIF(SUM(ft2.P_endingInning-ft2.P_startingInning),0) as P_IP,
NULLIF(SUM(ft2.P_Hit_by_Pitch),0) AS P_Hit_By_Pitch,
NULLIF(SUM(ft2.P_Home_Run),0) AS P_Home_Run,
NULLIF(AVG(ft2.P_Home_Run),0) AS P_Avg_Home_Run,
NULLIF(SUM(ft2.P_Strikeout),0) AS P_K,
NULLIF(9*(AVG(ft2.away_runs)/(ft2.P_endingInning-ft2.P_startingInning)),0) AS ERA
FROM feature_temp ft1
JOIN team t ON ft1.team_id = t.team_id
JOIN game g1 ON g1.game_id = ft1.game_id and g1.type="R"
JOIN feature_temp ft2 ON ft1.team_id = ft2.team_id
JOIN game g2 ON g2.game_id = ft2.game_id and g2.type="R" AND g2.local_date < g1.local_date
AND g2.local_date >= date_add(g1.local_date, INTERVAL - 100 day)
GROUP BY ft1.team_id, ft1.game_id, g1.local_date
ORDER BY ft1.team_id,g1.local_date;
CREATE UNIQUE INDEX rolling_100_days_idx ON rolling_100_days(team_id, game_id);


SELECT * FROM rolling_100_days;

#DICE = 3.00+((13*HR)+(3*(BB+HBP))-2K)/IP
DROP TABLE IF EXISTS feature_ratio;
CREATE TABLE IF NOT EXISTS feature_ratio
SELECT
    g.game_id,
    g.home_team_id,
    g.away_team_id,
    ROUND(((rdh.Hit / rdh.atBat) / (rda.Hit / rda.atBat)),5) AS BA_Ratio,
    ROUND(((rdh.atBat/rdh.Home_Run)/(rda.atBat/rda.Home_Run)),5) AS A_HR_Ratio,
    ROUND((rdh.Hit_By_Pitch / nullif(rda.Hit_By_Pitch,0)),5) AS HBP_Ratio,
    ROUND(((nullif(rdh.Home_Run,0) / rdh.Hit) / nullif((nullif(rda.Home_Run,0) / rda.Hit),0)),5) AS HRH_Ratio,
	ROUND((((rdh.TB-rdh.Hit)/rdh.atBat)/nullif(((rda.TB-rda.Hit)/rda.atBat),0)),5) AS ISO_Ratio,
	ROUND((((rdh.TB) / rdh.atBat) / ((rda.TB) / rda.atBat)),5) AS SLG_Ratio,
    ROUND((((rdh.Hit + rdh.BB + rdh.Hit_By_Pitch) / (rdh.atBat + rdh.BB + rdh.Hit_By_Pitch + rdh.Sac_Fly))
    / ((rda.Hit + rda.BB + rda.Hit_By_Pitch) / (rda.atBat + rda.BB + rda.Hit_By_Pitch + rda.Sac_Fly))),5) AS OBP_Ratio,
    ROUND((rdh.`2B`+rdh.`3B`+rdh.Home_Run)/(rda.`2B`+rda.`3B`+rda.Home_Run),5) AS XBH_Ratio,
    ROUND((rdh.TB / rda.TB),5) AS TB_Ratio,
    ROUND((rdh.TOB / rda.TOB),5) AS TOB_Ratio,
    ROUND((rdh.P_IP/rda.P_IP),5) AS IP_Ratio,
    ROUND(((((13*rdh.P_Home_Run)+(3*rdh.P_BB)-(2*rdh.P_K))/(rdh.P_IP+0.0001) )/(((13*rda.P_Home_Run+0.0001)+(3*rda.P_BB)-(2*rda.P_K))/(rda.P_IP+0.0001))),3) AS FIP_Ratio,
    ROUND(((9*(rdh.P_BB/rdh.P_IP))/(9*(rda.P_BB/rda.P_IP))),5) AS BB9_Ratio,
    ROUND(((9*(rdh.P_K/rdh.P_IP))/(9*(rda.P_K/rda.P_IP))),5) AS K9_Ratio,
    ROUND((rdh.ERA / nullif(rda.ERA,0)),5) AS ERA_Ratio,
    ROUND((3.00+((13*rdh.P_Home_Run)+(3*(rdh.P_BB+rdh.P_Hit_By_Pitch))-(2*rdh.P_K))/rdh.P_IP)/(3.00+((13*rda.P_Home_Run)+(3*(rdh.BB+rda.P_Hit_By_Pitch))-(2*rda.P_K))/rdh.P_IP),5) AS DICE_Ratio,
    ROUND(((rdh.P_Avg_Home_Run)/rdh.P_IP)/NULLIF(((rda.P_Avg_Home_Run)/rda.P_IP),0),5) AS HR9_Ratio,
    ROUND((((rdh.P_Home_Run + rdh.P_BB) / rdh.P_IP) / ((rda.P_Home_Run + rda.P_BB) / rda.P_IP)),5) AS WHIP_Ratio,
    ROUND(((rdh.P_K/rdh.P_BB)/(rda.P_K/rda.P_BB)),5) AS KBB_Ratio,
    CASE WHEN b.away_runs < b.home_runs THEN 1
    WHEN b.away_runs > b.home_runs THEN 0
    ELSE 0 END AS home_team_wins
FROM
    game g
        JOIN
    rolling_100_days rdh ON g.game_id = rdh.game_id
        AND g.home_team_id = rdh.team_id
        JOIN
    rolling_100_days rda ON g.game_id = rda.game_id
        AND g.away_team_id = rda.team_id
        JOIN boxscore b ON b.game_id = g.game_id;

SELECT * FROM feature_ratio;



DROP TABLE IF EXISTS feature_diff;
CREATE TABLE IF NOT EXISTS feature_diff
SELECT
    g.game_id,
    g.home_team_id,
    g.away_team_id,
	ROUND(((rdh.Hit / rdh.atBat) - (rda.Hit / rda.atBat)),5) AS BA_diff,
	ROUND(((rdh.atBat/rdh.Home_Run)-(rda.atBat/rda.Home_Run)),5) AS A_HR_diff,
	ROUND((rdh.Hit_By_Pitch - nullif(rda.Hit_By_Pitch,0)),5) AS HBP_diff,
	ROUND(((nullif(rdh.Home_Run,0) / rdh.Hit) - nullif((nullif(rda.Home_Run,0) / rda.Hit),0)),5) AS HRH_diff,
	ROUND((((rdh.TB-rdh.Hit)/rdh.atBat)-nullif(((rda.TB-rda.Hit)/rda.atBat),0)),5) AS ISO_diff,
	ROUND((((rdh.TB) / rdh.atBat) - ((rda.TB) / rda.atBat)),5) AS SLG_diff,
	ROUND((((rdh.Hit + rdh.BB + rdh.Hit_By_Pitch) / (rdh.atBat + rdh.BB + rdh.Hit_By_Pitch + rdh.Sac_Fly))
	 - ((rda.Hit + rda.BB + rda.Hit_By_Pitch) / (rda.atBat + rda.BB + rda.Hit_By_Pitch + rda.Sac_Fly))),5) AS OBP_diff,
	ROUND((rdh.`2B`+rdh.`3B`+rdh.Home_Run)-(rda.`2B`+rda.`3B`+rda.Home_Run),5) AS XBH_diff,
	ROUND((rdh.TB - rda.TB),5) AS TB_diff,
	ROUND((rdh.TOB - rda.TOB),5) AS TOB_diff,
	ROUND((rdh.P_IP-rda.P_IP),5) AS IP_diff,
	ROUND(((((13*rdh.P_Home_Run)+(3*rdh.P_BB)-(2*rdh.P_K))/rdh.P_IP )-(((13*rda.P_Home_Run)+(3*rda.P_BB)-(2*rda.P_K))/rda.P_IP )),3) AS FIP_diff,
	ROUND(((9*(rdh.P_BB/rdh.P_IP))-(9*(rda.P_BB/rda.P_IP))),5) AS BB9_diff,
	ROUND(((9*(rdh.P_K/rdh.P_IP))-(9*(rda.P_K/rda.P_IP))),5) AS K9_diff,
	ROUND((rdh.ERA - nullif(rda.ERA,0)),5) AS ERA_diff,
	ROUND((3.00+((13*rdh.P_Home_Run)+(3*(rdh.P_BB+rdh.P_Hit_By_Pitch))-(2*rdh.P_K))/rdh.P_IP)-(3.00+((13*rda.P_Home_Run)+(3*(rdh.BB+rda.P_Hit_By_Pitch))-(2*rda.P_K))/rdh.P_IP),5) AS DICE_diff,
	ROUND((((rdh.P_Avg_Home_Run)/rdh.P_IP)-((rda.P_Avg_Home_Run)/rda.P_IP)),5) AS HR9_diff,
	ROUND((((rdh.P_Home_Run + rdh.P_BB) / rdh.P_IP) - ((rda.P_Home_Run + rda.P_BB) / rda.P_IP)),5) AS WHIP_diff,
	ROUND(((rdh.P_K/rdh.P_BB)-(rda.P_K/rda.P_BB)),5) AS KBB_diff,
    CASE WHEN b.away_runs < b.home_runs THEN 1
    WHEN b.away_runs > b.home_runs THEN 0
    ELSE 0 END AS home_team_wins
FROM
    game g
        JOIN
    rolling_100_days rdh ON g.game_id = rdh.game_id
        AND g.home_team_id = rdh.team_id
        JOIN
    rolling_100_days rda ON g.game_id = rda.game_id
        AND g.away_team_id = rda.team_id
        JOIN boxscore b ON b.game_id = g.game_id;

SELECT * FROM feature_diff;


DROP TABLE IF EXISTS feature_per;
CREATE TABLE IF NOT EXISTS feature_per
SELECT
    g.game_id,
    g.home_team_id,
    g.away_team_id,
	ROUND((((rdh.Hit / rdh.atBat) - (rda.Hit / rda.atBat))/((rdh.Hit / rdh.atBat)+0.0001))*100,5) AS BA_per,
	ROUND((((rdh.atBat/rdh.Home_Run)-(rda.atBat/rda.Home_Run))/((rdh.atBat/rdh.Home_Run)+0.0001))*100,5) AS A_HR_per,
	ROUND(((rdh.Hit_By_Pitch - nullif(rda.Hit_By_Pitch,0))/rdh.Hit_By_Pitch+0.0001)*100,5) AS HBP_per,
	ROUND((((nullif(rdh.Home_Run,0) / rdh.Hit) - nullif((nullif(rda.Home_Run,0) / rda.Hit),0))/(nullif(rdh.Home_Run,0) / rdh.Hit))*100,5) AS HRH_per,
	ROUND(((((rdh.TB-rdh.Hit)/rdh.atBat)-nullif(((rda.TB-rda.Hit)/rda.atBat),0))/(((rdh.TB-rdh.Hit)/(rdh.atBat+0.0001)+0.0001)))*100,5) AS ISO_per,
	ROUND(((((rdh.TB) / rdh.atBat) - ((rda.TB) / rda.atBat))/((rdh.TB) / rdh.atBat))*100,5) AS SLG_per,
	ROUND(((((rdh.Hit + rdh.BB + rdh.Hit_By_Pitch) / (rdh.atBat + rdh.BB + rdh.Hit_By_Pitch + rdh.Sac_Fly))
	- ((rda.Hit + rda.BB + rda.Hit_By_Pitch) / (rda.atBat + rda.BB + rda.Hit_By_Pitch + rda.Sac_Fly)))/((rdh.Hit + rdh.BB + rdh.Hit_By_Pitch) / (rdh.atBat + rdh.BB + rdh.Hit_By_Pitch + rdh.Sac_Fly)))*100,5) AS OBP_per,
	ROUND((((rdh.`2B`+rdh.`3B`+rdh.Home_Run)-(rda.`2B`+rda.`3B`+rda.Home_Run))
	/(rdh.`2B`+rdh.`3B`+rdh.Home_Run))*100,5) AS XBH_diff,
	ROUND(((rdh.TB - rda.TB)/rdh.TB+0.0001)*100,3) AS TB_per,
	ROUND(((rdh.TOB - rda.TOB)/rdh.TOB+0.0001)*100,5) AS TOB_per,
	ROUND(((rdh.P_IP-rda.P_IP)/rdh.P_IP+0.0001)*100,5) AS IP_per,
	ROUND(((((((13*rdh.P_Home_Run)+(3*rdh.P_BB)-(2*rdh.P_K))/(rdh.P_IP+0.0001))-
	(((13*rda.P_Home_Run)+(3*rda.P_BB)-(2*rda.P_K))/(rda.P_IP+0.0001))))/
	(((13*rdh.P_Home_Run+0.0001)+(3*rdh.P_BB)-(2*rdh.P_K))/(rdh.P_IP+0.0001)))*100,5) AS FIP_per,
	ROUND((((9*(rdh.P_BB/rdh.P_IP))-(9*(rda.P_BB/rda.P_IP)))/(9*(rdh.P_BB/rdh.P_IP)))*100,5) AS BB9_per,
	ROUND((((9*(rdh.P_K/rdh.P_IP))-(9*(rda.P_K/rda.P_IP)))/(9*(rdh.P_K/rdh.P_IP)))*100,5) AS K9_per,
	ROUND(((rdh.ERA - nullif(rda.ERA,0))/(rdh.ERA+0.0001))*100,5) AS ERA_per,
	ROUND((((3.00+((13*rdh.P_Home_Run)+(3*(rdh.P_BB+rdh.P_Hit_By_Pitch))-(2*rdh.P_K))/rdh.P_IP)-
	(3.00+((13*rda.P_Home_Run)+(3*(rdh.BB+rda.P_Hit_By_Pitch))-(2*rda.P_K))/rdh.P_IP))/(
	3.00+((13*rdh.P_Home_Run)+(3*(rdh.P_BB+rdh.P_Hit_By_Pitch))-(2*rdh.P_K))/rdh.P_IP))*100,5) AS DICE_per,
	ROUND(((((rdh.P_Avg_Home_Run)/rdh.P_IP)-((rda.P_Avg_Home_Run)/rda.P_IP))/
	((rdh.P_Avg_Home_Run)/rdh.P_IP))*100,5) AS HR9_per,
	ROUND(((((rdh.P_Home_Run + rdh.P_BB) / rdh.P_IP)-((rda.P_Home_Run + rda.P_BB) / rda.P_IP))
	/((rdh.P_Home_Run + rdh.P_BB) / (rdh.P_IP+0.0001)))*100,5) AS WHIP_per,
	ROUND((((rdh.P_K/rdh.P_BB)/(rda.P_K/rda.P_BB))/(rdh.P_K/rdh.P_BB))*100,5) AS KBB_per,
	CASE WHEN b.away_runs < b.home_runs THEN 1
	WHEN b.away_runs > b.home_runs THEN 0
	ELSE 0 END AS home_team_wins
	FROM game g
        JOIN
    rolling_100_days rdh ON g.game_id = rdh.game_id
        AND g.home_team_id = rdh.team_id
        JOIN
    rolling_100_days rda ON g.game_id = rda.game_id
        AND g.away_team_id = rda.team_id
        JOIN boxscore b ON b.game_id = g.game_id;


SELECT * FROM feature_per;
