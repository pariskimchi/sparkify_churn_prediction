import configparser


# CONFIG
config = configparser.ConfigParser()
config.read('./dwh.cfg')

ARN = config.get("IAM_ROLE","ARN")
LOG_DATA = config.get("S3","LOG_DATA")
LOG_JSON_PATH = config.get("S3","LOG_JSONPATH")
SONG_DATA = config.get("S3","SONG_DATA")

WEEK_SUMMARY_DATA = config.get("S3","WEEK_SUMMARY_DATA")

# DROP TABLES
staging_week_summary_drop = "DROP TABLE IF EXISTS staging_week_summary"

# CREATE TABLES
staging_week_summary_table_create = (
    """
    CREATE TABLE IF NOT EXISTS staging_week_summary (
        userId varchar,
        level varchar,
        last_interaction date,
        week_code bigint,
        day_from_reg int,
        churn_service bigint,
        churn_paid bigint,
        count_login bigint,
        time_inter_login float,
        count_song bigint,
        count_song_weekday bigint,
        count_song_weekend bigint,
        count_ads bigint,
        ratio_song_ads float,
        count_repeat bigint,
        repeat_ratio float,
        week_avg_listen_item_session float,
        total_action_week bigint,
        avg_num_action_session  float

    )
    """
)

# STAGING TABLES  COPY

staging_week_summary_copy = (
    """
    COPY staging_week_summary
    FROM {}
    iam_role {}
    FORMAT AS PARQUET;
    """
).format(WEEK_SUMMARY_DATA,ARN)



# INSERT 

week_summary_table_insert = (
    """
    INSERT INTO week_summary (userId,level,last_interaction,week_code,
    day_from_reg,churn_service,churn_paid,count_login,time_inter_login,count_song,count_song_weekday,count_song_weekend,count_ads,ratio_song_ads,count_repeat,repeat_ratio,
    week_avg_listen_item_session,total_action_week,avg_num_action_session)
    SELECT * FROM stage_week_summary
    
    """
)

# QUERY LISTS
drop_table_queries = [staging_week_summary_drop]
copy_table_queries = [staging_week_summary_copy]
create_table_queries = [staging_week_summary_table_create]
