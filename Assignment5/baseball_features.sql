# Features Selected:
# Batting Statistics
# 1. BA =  Hit/atbat
# 2. A_HR = atBat/Home_RUN
# 3. HBP = Hit by Pitch
# 4. HRH = Home Run per Hit
# 5. ISO = Isolated power (TB-B/atBAt)
# 6. SLG = Slugging Average (TB/atBAt)
# 7. OBP = on base percentage (H+BB+HBP)/(atBAt+BB+HBP+Sac_Fly)
# 8. TB = Total Bases [H + 2B + (2 × 3B) + (3 × HR)] or [1B + (2 × 2B) + (3 × 3B) + (4 × HR)]
# 9. TOB = Times on base (H + BB + HBP)
# Pitching Statistics (Got idea of good pitching statistics from wiki page)
# 10. IP = Inning Pitched (endingInning-startingInning)
# 11. ERA = earned run average 9*(earned run allowed/IP)
# 12. WHIP = walks plus hits per inning pitched
# For calculating statistical feature Home_team_wins response:
#->we can calculate feature ratio between  home and away
#-> Calculate Feature difference between home and away
#-> calculate difference percentage on the basis of home feature. [Feature_diff/Feature_home]*100
# I tried for ratio and diff, By using ratio I am getting better result so for modelling I used ratio code.
# In my sql code, I have calculated feature_ratio and feature_diff tables.
#used ref https://fisher.wharton.upenn.edu/wp-content/uploads/2020/09/Thesis_Andrew-Cui.pdf and
# towardsdatascience.com



USE baseball;

DROP TEMPORARY TABLE IF EXISTS team_pitch_temp;
CREATE TEMPORARY TABLE team_pitch_temp ENGINE=MEMORY AS
SELECT tbc.*,
pc.startingPitcher,
SUM(pc.startingInning) AS startingInning ,
SUM(pc.endingInning) AS endingInning
FROM team_pitching_counts tbc
JOIN pitcher_counts pc ON pc.game_id = tbc.game_id
AND pc.team_id = tbc.team_id
GROUP BY pc.team_id,pc.game_id;
CREATE INDEX team_pitch_temp_idx ON team_pitch_temp (team_id, game_id);

SELECT * FROM team_pitch_temp;

ALTER TABLE team_pitch_temp
CHANGE COLUMN win P_win INT NULL DEFAULT 0 ,
CHANGE COLUMN atBat P_atBat INT NULL DEFAULT 0 ,
CHANGE COLUMN Hit P_Hit INT NULL DEFAULT 0 ,
CHANGE COLUMN Fly_Out P_Fly_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Hit_By_Pitch P_Hit_By_Pitch FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Home_Run P_Home_Run FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Strikeout P_Strikeout FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Walk P_Walk FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN startingInning P_startingInning DOUBLE NULL DEFAULT 0 ,
CHANGE COLUMN endingInning P_endingInning DOUBLE NULL DEFAULT 0 ;


DROP TABLE IF EXISTS feature_temp;
CREATE TABLE IF NOT EXISTS feature_temp
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
NULLIF(SUM(ft2.Single+2*ft2.Double+3*ft2.Triple+4*ft2.Home_Run),0) AS TB,
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
JOIN game g1 ON g1.game_id = ft1.game_id
JOIN feature_temp ft2 ON ft1.team_id = ft2.team_id
JOIN game g2 ON g2.game_id = ft2.game_id AND g2.local_date < g1.local_date
AND g2.local_date >= date_add(g1.local_date, INTERVAL - 100 day)
GROUP BY ft1.team_id, ft1.game_id, g1.local_date
ORDER BY ft1.team_id,g1.local_date;
CREATE UNIQUE INDEX rolling_100_days_idx ON rolling_100_days(team_id, game_id);


SELECT * FROM rolling_100_days;


DROP TABLE IF EXISTS feature_ratio;
CREATE TABLE IF NOT EXISTS feature_ratio
SELECT 
    g.game_id,
    g.home_team_id,
    g.away_team_id,
    ROUND(((rdh.Hit / rdh.atBat) / (rda.Hit / rda.atBat)),3) AS BA_Ratio,
    ROUND(((rdh.atBat/rdh.Home_Run)/(rda.atBat/rda.Home_Run)),3) AS A_HR_Ratio,
    ROUND((rdh.Hit_By_Pitch / NULLIF(rda.Hit_By_Pitch,0)),3) AS HBP_Ratio,
    ROUND(((NULLIF(rdh.Home_Run,0) / rdh.Hit) / NULLIF((NULLIF(rda.Home_Run,0) / rda.Hit),0)),3) AS HRH_Ratio,
    ROUND((((rdh.TB-rdh.B)/rdh.atBat)/NULLIF(((rda.TB-rda.B)/rda.atBat),0)),3) AS ISO_Ratio,
    ROUND((((rdh.TB) / rdh.atBat) / ((rda.TB) / rda.atBat)),3) AS SLG_Ratio,
    ROUND((((rdh.Hit + rdh.BB + rdh.Hit_By_Pitch) / (rdh.atBat + rdh.BB + rdh.Hit_By_Pitch + rdh.Sac_Fly))
    / ((rda.Hit + rda.BB + rda.Hit_By_Pitch) / (rda.atBat + rda.BB + rda.Hit_By_Pitch + rda.Sac_Fly))),3) AS OBP_Ratio,
    ROUND((rdh.TB / rda.TB),3) AS TB_Ratio,
    ROUND((rdh.TOB / rda.TOB),3) AS TOB_Ratio,
    ROUND((rdh.P_IP/rda.P_IP),3) AS IP_Ratio,ROUND((rdh.ERA / NULLIF(rda.ERA,0)),3) AS P_ERA_Ratio,
    ROUND((((rdh.P_Home_Run + rdh.P_BB) / rdh.P_IP) / ((rda.P_Home_Run + rda.P_BB) / rda.P_IP)),3) AS P_WHIP_Ratio,  
    CASE WHEN b.away_runs < b.home_runs THEN 1
    WHEN b.away_runs > b.home_runs THEN 0
    ELSE 0 END AS home_team_wins
FROM
    game g
        JOIN rolling_100_days rdh ON g.game_id = rdh.game_id
        AND g.home_team_id = rdh.team_id
        JOIN rolling_100_days rda ON g.game_id = rda.game_id
        AND g.away_team_id = rda.team_id
        JOIN boxscore b ON b.game_id = g.game_id;
       
SELECT * FROM feature_ratio;



DROP TABLE IF EXISTS feature_diff;
CREATE TABLE IF NOT EXISTS feature_diff
SELECT 
    g.game_id,
    g.home_team_id,
    g.away_team_id,
    ROUND(((rdh.Hit / rdh.atBat) - (rda.Hit / rda.atBat)),3) AS BA_diff,
    ROUND(((rdh.atBat/rdh.Home_Run)-(rda.atBat/rda.Home_Run)),3) AS A_HR_diff,
    ROUND((rdh.Hit_By_Pitch - NULLIF(rda.Hit_By_Pitch,0)),3) AS HBP_diff,
    ROUND(((NULLIF(rdh.Home_Run,0) / rdh.Hit) - NULLIF((NULLIF(rda.Home_Run,0) / rda.Hit),0)),3) AS HRH_diff,
    ROUND((((rdh.TB-rdh.B)/rdh.atBat)-NULLIF(((rda.TB-rda.B)/rda.atBat),0)),3) AS ISO_diff,
    ROUND((((rdh.TB) / rdh.atBat) - ((rda.TB) / rda.atBat)),3) AS SLG_diff,
    ROUND((((rdh.Hit + rdh.BB + rdh.Hit_By_Pitch) / (rdh.atBat + rdh.BB + rdh.Hit_By_Pitch + rdh.Sac_Fly))
    - ((rda.Hit + rda.BB + rda.Hit_By_Pitch) / (rda.atBat + rda.BB + rda.Hit_By_Pitch + rda.Sac_Fly))),3) AS OBP_diff,
    ROUND((rdh.TB - rda.TB),3) AS TB_diff,
    ROUND((rdh.TOB - rda.TOB),3) AS TOB_diff,
    ROUND((rdh.P_IP-rda.P_IP),3) AS IP_diff,
    ROUND((rdh.ERA - NULLIF(rda.ERA,0)),3) AS P_ERA_diff,
    ROUND((((rdh.P_Home_Run + rdh.P_BB) / rdh.P_IP) - ((rda.P_Home_Run + rda.P_BB) / rda.P_IP)),3) AS P_WHIP_diff, 
    CASE WHEN b.away_runs < b.home_runs THEN 1
    WHEN b.away_runs > b.home_runs THEN 0
    ELSE 0 END AS home_team_wins
FROM
    game g
        JOIN rolling_100_days rdh ON g.game_id = rdh.game_id
        AND g.home_team_id = rdh.team_id
        JOIN rolling_100_days rda ON g.game_id = rda.game_id
        AND g.away_team_id = rda.team_id
        JOIN boxscore b ON b.game_id = g.game_id;
       
SELECT * FROM feature_diff;







