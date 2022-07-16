from asyncio.log import logger
import configparser
from distutils.log import error
import os
import logging

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

import pyspark 
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
from pyspark.sql.functions import udf, isnan, col, upper 
# from pyspark.sql.functions import 
from pyspark.sql import Window 
from datetime import datetime, timedelta 


# feature engineering 
from pyspark.ml import Pipeline 
from pyspark.ml.feature import MinMaxScaler,  StringIndexer, VectorAssembler 
from pyspark.sql.types import StructField, StructType, IntegerType, FloatType, StringType, DateType

# feature engineering 
# from pyspark.ml import Pipeline 
# from pyspark.ml.feature import MinMaxScaler,  StringIndexer, VectorAssembler 
# from pyspark.sql.types import StructField, StructType, IntegerType, FloatType, StringType 

# # modeling 
# from pyspark.sql.functions import split 
# from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier 
# from pyspark.ml.evaluation  import MulticlassClassificationEvaluator 
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def load_data_format(spark_session, data_path):
    """
        Load a dataset file from the spark session corresponding to each data format    
    """

    if data_path.contains("/"):
        data_format = data_path.split("/")[-1]\
                .split(".")[-1]
    else:
        data_format = data_path.split(".")[-1]

    if data_format == 'json':
        return spark_session.read.json(data_path)
    elif data_format == 'csv':
        return spark_session.read.csv(data_path)
    elif data_format == 'parquet':
        return spark_session.read.parquet(data_path)

    else:
        raise TypeError


# drop null values 
def drop_na(df,output_path):
    """
        drop null value on userId, sessionId
    """
    
    # define filter condition 
    filter_user_id = (df['userId'] != "") & (df['userId'].isNotNull())& (~isnan(df['userId']))
    filter_session_id = (df['sessionId'].isNotNull()) & (~isnan(df['sessionId']))
    
    df_clean = df.filter(filter_user_id).filter(filter_session_id)
    
    # write drop_df
    df_clean.write.parquet(output_path+"/lake_cleaned"+"/clean_drop",mode="overwrite")
    
    
    return df_clean

# convert ts to timestamp, an dcreate date, year, month, day, hour, weekday, weekofyear
def convert_ts(df,output_path):
    """
        Convert timestamp column into serveral time columns 
    """
    ts_cols = ['userId','level','ts']
    
    ts = (F.col('ts')/1000).cast('timestamp')
    
    
    df_clean = df.select(ts_cols)\
        .withColumn('date',F.date_format(ts,format='yyyy-MM-dd'))\
        .withColumn('date',F.to_date(F.col('date'),'yyyy-MM-dd'))\
        .withColumn('year',F.year(F.col('date')))\
        .withColumn('month',F.month(F.col('date')))\
        .withColumn('day',F.dayofmonth(F.col('date')))\
        .withColumn('hour', F.hour(ts))\
        .withColumn('dayofweek',F.dayofweek(F.col('date')))\
        .withColumn('weekofyear',F.weekofyear(F.col('date')))
    
    # write new convert_ts columns to lake 
    # convert_ts_columns 
    df_clean.write.parquet(output_path+"/lake_cleaned"+"/clean_ts",mode="overwrite")
    
    
    # return df_clean

# convert ts to timestamp, an dcreate date, year, month, day, hour, weekday, weekofyear
def convert_registration(df,output_path):
    """
        Convert registration column into serveral time columns 
    """
    regi_ts = (F.col('registration')/1000).cast('timestamp')
    
    regi_cols = ['userId','level','registration']
        
    
    df_regi = df.select(regi_cols)\
        .distinct()\
        .withColumn('regi_date',F.date_format(regi_ts,format='yyyy-MM-dd'))\
            .withColumn('regi_date',F.to_date(F.col('regi_date'),'yyyy-MM-dd'))\
            .withColumn('regi_year',F.year(F.col('regi_date')))\
            .withColumn('regi_month',F.month(F.col('regi_date')))\
            .withColumn('regi_day',F.dayofmonth(F.col('regi_date')))\
            .withColumn('regi_hour', F.hour(regi_ts))\
            .withColumn('regi_dayofweek',F.dayofweek(F.col('regi_date')))\
            .withColumn('regi_weekofyear',F.weekofyear(F.col('regi_date')))
    
    # write parquet
    df_regi.write.parquet(output_path+"/lake_cleaned"+"/clean_regi/",mode="overwrite")
    
    # return df_regi
    
# extract location_state
def extract_location_state(df,output_path):
    """
        splits the location column to extract state
    """
    location_cols = ['userId','level','location']
    
    df_extract =  df.select(location_cols)\
        .distinct()\
        .withColumn('state',F.split(F.col('location'),", ").getItem(1))
    
    # write parquet 
    df_extract.write.parquet(output_path+"/lake_cleaned"+"/clean_location",mode="overwrite")
    
    
    # return df_extract

def set_churn_label(df,output_path):
    """
        Add churn_label, upgrade, downgrade, cancelled
    """

    def add_cancelled():
        col = F.when(F.col('page')=='Cancellation Confirmation',F.lit(1))\
                .otherwise(F.lit(0))

        return col

    def add_downgraded():
        col = F.when(F.col('page')=='Submit Downgrade',F.lit(1))\
                .otherwise(F.lit(0))

        return col

    def add_upgraded():
        col = F.when(F.col('page')=='Submit Upgrade',F.lit(1))\
                .otherwise(F.lit(0))

        return col
   
    
    # select userId, level page
    df_flag = df.select(['userId','level','page','ts'])\
        .withColumn('cancelled',add_cancelled())\
    .withColumn('downgraded',add_downgraded())\
    .withColumn('upgraded',add_upgraded())
    
    #set windowval and create phase columns 
    windowval = Window.partitionBy(['userId','level']).orderBy(F.desc('ts')).rangeBetween(Window.unboundedPreceding,0)
    df_phase = df_flag.withColumn('phase_downgrade',F.sum('downgraded').over(windowval))\
        .withColumn('phase_upgrade',F.sum('upgraded').over(windowval))
    
    # phase_cancel
    window_cancel = Window.partitionBy(['userId']).orderBy(F.desc('ts')).rangeBetween(Window.unboundedPreceding,0)
    df_phase = df_phase.withColumn('phase_cancel',F.sum('cancelled').over(window_cancel))
    
    # create total churn num
    df_churn = df_phase.withColumn('churn_paid',F.sum('downgraded').over(Window.partitionBy('userId')))\
    .withColumn('churn_service',F.sum('cancelled').over(Window.partitionBy('userId')))
    
#     df_churn_merge = df.join(df_churn,on=['userId','level','page','ts'],how='left')
    
    # write parquet 
    df_churn.write.parquet(output_path+"/lake_cleaned"+"/clean_churn_label",mode="overwrite")

    # return df_churn



def processing_cleaning_df(spark,df,output_path):
    """
        
    """

    logger.info("Start Processing Drop_na")
    df_clean = drop_na(df,output_path)
    logger.info("Completed Processing Drop_na df")

    # load df_clean or not 
    df_clean = spark.read.parquet(output_path+"/lake_cleaned/clean_drop")
    # processing convert ts 
    logger.info("Start Processing convert_ts")
    convert_ts(df_clean,output_path)
    logger.info("Complete Processing convert_ts")

    # processing convert regi
    logger.info("Start Processing convert_regi")
    convert_registration(df_clean,output_path)
    logger.info("Complete Processing convert_regi")

    # processing extract location
    logger.info("Start Processing location")
    extract_location_state(df_clean,output_path)
    logger.info("Complete Proessing location")

    # processing set_churn_label function
    logger.info("Start Processing churn_label")
    set_churn_label(df_clean,output_path)
    logger.info("Complete Processing churn_label")

    logger.info("Processing Cleaning df Completed!")


# # prepared df
def save_cleaned_df(spark,output_path):
    """
        Combine cleaned dataframe into full dataframe cleaned on AWS s3 as Data Lake
    """

    df_after_drop = spark.read.parquet(output_path+"/lake_cleaned"+"/clean_drop")
    df_ts = spark.read.parquet(output_path+"/lake_cleaned"+"/clean_ts")
    df_regi = spark.read.parquet(output_path+"/lake_cleaned"+"/clean_regi")
    df_location = spark.read.parquet(output_path+"/lake_cleaned"+"/clean_location")
    df_churn_label = spark.read.parquet(output_path+"/lake_cleaned"+"/clean_churn_label")
    
    # all join??
    df_clean = df_after_drop.join(df_ts,on=['userId','level','ts'],how='left')\
            .join(df_regi,on=['userId','level','registration'],how='inner')\
            .join(df_location,on=['userId','level','location'],how='inner')\
            .join(df_churn_label,on=['userId','level','ts','page'],how='left')

    # all joined write?
    df_clean.write.parquet(output_path+"/lake_cleaned_df_full",mode="overwrite")
    
    logger.info("Complete Writing parquet all joined df_clean")


def load_clean_df(spark,clean_df_path):
    """
        Load cleaned dataframe from s3 data lake where cleaned dataframe is stored
    """

    df_clean = spark.read.parquet(clean_df_path)
    
    return df_clean

def processing_calendar_week(df_clean,output_path):
    """
        process to build week, calendar table
    
    """
    
    # create df_calendar
    calendar_cols = ['date','year','month','day','weekofyear']

    df_calendar = df_clean.select(calendar_cols).distinct()\
        .orderBy(F.asc('date'))\
        .withColumn('date',F.col('date').cast('string'))\
        .withColumn('date_code',F.monotonically_increasing_id()+1)
    
    # create df_week
    week_cols = ['year','month','weekofyear']
    df_week = df_calendar.select(week_cols)\
        .distinct().orderBy(F.asc('weekofyear'))\
        .withColumn('week_code',F.monotonically_increasing_id()+1)
    
    # create df_merge_calendar 
    df_merge_calendar = df_calendar.join(df_week,on=['year','month','weekofyear'],how='left')
    
    # write df_week as parquet to 'dim_week'
    df_week.write.parquet(output_path+"/dim_week",mode="overwrite")
    
    # write df_merge_calendar as parquet to 'dim_calendar'
    df_merge_calendar.write.parquet(output_path+"/dim_calendar",mode="overwrite")
    
    return df_merge_calendar

def get_week_from_df(spark,df, output_path,start_date, end_date):
    """
        Function to extract the rows corresponding to a given week
    """

    def weekday_weekend():
        col = F.when(F.col('dayofweek').isin(['1','2','3','4','5']),F.lit(1))\
                .otherwise(F.lit(0))
        
        return col

    def is_ads():
        col = F.when(F.col('page')=='Roll Advert',F.lit(1))\
                .otherwise(F.lit(0))

        return col
    # filter by week day_start, week_day_last 
    filter_week = (F.col('date')>=start_date) &(F.col('date')<=end_date)
    
    # apply filter_week 
    df_week = df.filter(filter_week).orderBy(F.desc('ts'))\
        .withColumn('is_weekday',weekday_weekend())\
        .withColumn('is_ads',is_ads())
    
    # load calendar df
    df_week_calendar = spark.read.parquet(output_path+"/dim_calendar")
    
    df_week = df_week.join(df_week_calendar.select(['date','week_code']),on=['date'],how='left')
    
    return df_week

def processing_churn_label(df_clean,output_path, start_date, end_date):
    """
        build churn lake 
    """
    churn_label_cols = ['userId','churn_service','churn_paid']

    df_churn = df_clean.select(churn_label_cols).distinct()
    
    df_churn.write.parquet(output_path+"/dim_churn_label/{}_{}".format(start_date,end_date),mode="overwrite")

    return df_churn

def processing_login(df_week,output_path,start_date, end_date):
    """
        build login table
    """
    df_count_login = df_week.groupby(['userId','week_code'])\
    .agg(F.countDistinct('sessionId').alias('count_login'))\
    .distinct()
    
    # df_delta_login
    window_login = Window.partitionBy(['userId']).orderBy(F.desc('date'))

    # add new column next date
    df_delta = df_week.withColumn('next_date',F.lag('date',1).over(window_login))

    # caculate delta between two login 
    df_delta = df_delta.withColumn('delta_time',F.datediff(F.col('next_date'),F.col('date')))

    delta_cols = ['userId','level','week_code','delta_time']

    df_delta_login = df_delta.select(delta_cols)\
                .filter(df_delta['delta_time'] != 0)


    distinct_session = df_week.groupby('userId','week_code').agg(F.countDistinct('sessionId').alias('user_num_session'))

    # join: df_delta_login + distinct_session
    df_delta_login = df_delta_login.join(distinct_session, on=['userId','week_code'], how='inner')

    # calculate avg delta time , total_avg_delta
    df_delta_login = df_delta_login.withColumn('avg_delta',F.col('delta_time')/F.col('user_num_session'))\
        .withColumn('total_avg_delta',F.sum('avg_delta').over(Window.partitionBy(['userId','week_code'])))


    # group by , relevant data 
    df_delta_login = df_delta_login.groupby(['userId','level','week_code'])\
        .agg(F.max('total_avg_delta').alias('time_inter_login'))
    # fill na : 7 
    df_delta_login = df_delta_login.na.fill(0)

    df_merge_login = df_delta_login.join(df_count_login,on=['userId','week_code'],how='inner')
    
    df_merge_login = df_merge_login.withColumn('login_code',F.monotonically_increasing_id()+1)
    
    # write parquet 
    df_merge_login.write.parquet(output_path+"/dim_login/{}_{}".format(start_date,end_date),mode="overwrite")
    
    return df_merge_login

def processing_song_ads(df_week,output_path,start_date, end_date):
    """
        유저의 주간별 전체,평일,주말의 노래 시청 횟수 및 광고 횟수,
        노래당 광고 시청 비율 집계 및 테이블 생성
    """
    cols = ['userId','level','week_code']
    df_user = df_week.select(cols).distinct()
    
    song_week = df_week.groupby(cols)\
                .agg(F.count('song').alias('count_song'))

    song_weekday = df_week.filter(F.col('is_weekday')==1)\
        .groupby(cols)\
        .agg(F.countDistinct('song').alias('count_song_weekday'))

    song_weekend = df_week.filter(F.col('is_weekday')!=1)\
        .groupby(cols)\
        .agg(F.countDistinct('song').alias('count_song_weekend'))

    ads_week = df_week.groupby(cols)\
                .agg(F.sum('is_ads').alias('count_ads'))
    df_song_ads = df_user.join(song_week, on=cols, how='left')\
            .join(song_weekday, on=cols, how='left')\
            .join(song_weekend, on=cols, how='left')\
            .join(ads_week, on=cols, how='left')\
            .na.fill(0)\
            .withColumn('ratio_song_ads',F.round(F.col('count_ads')/F.col('count_song'),2))\
            .na.fill(0)\
        .withColumn('song_ads_code', F.monotonically_increasing_id()+1)

    df_song_ads.write.parquet(output_path+"/dim_song_ads/{}_{}".format(start_date, end_date),mode="overwrite")
    
    return df_song_ads

def processing_repeat(df_week,output_path,start_date, end_date):
    """
        유저의 주간 별 같은 노래의 반복 횟수 및 같은 아티스트의 반복횟수,
        같은 노래 반복 비율을 집계 하고 테이블 생성
    """
    repeat_cols = ['userId','level','week_code']

    df_repeat = df_week.select(['userId','level','week_code','song','artist']).groupby(repeat_cols)\
        .agg(F.count('song').alias('total_num_song'),\
            F.countDistinct('song').alias('unique_song'),\
            F.countDistinct('artist').alias('distinct_artist'))\
        .withColumn('count_repeat',F.col('total_num_song')-F.col('unique_song'))\
        .withColumn('repeat_ratio',F.round(1-(F.col('unique_song')/F.col('total_num_song'))))\
        .na.fill(0)\
        .withColumn('repeat_distinct_artist',F.round(F.col('distinct_artist')/F.col('total_num_song')))\
        .withColumn('repeat_code',F.monotonically_increasing_id()+1)
    
    df_repeat.write.parquet(output_path+"/dim_repeat/{}_{}".format(start_date, end_date),mode="overwrite")
    
    return df_repeat

def processing_listen(df_week,output_path,start_date, end_date):
    """
        유저의 주간별 노래 시청 시간 및 세션별 평균 시청시간 집계 후 
        테이블 생성
    """
    # window per user by desc timestmp
    window_user = Window.partitionBy("userId").orderBy(F.desc('ts'))
    #window per user, session
    window_session = Window.partitionBy(["userId","sessionId"]).orderBy("ts").rangeBetween(Window.unboundedPreceding,0)
    

    # add two new columns: next_ts, next_action
    listen_cols = ['userId','level','week_code','sessionId','itemInSession','ts','page','song']

    df_listen_session = df_week.select(listen_cols)\
        .withColumn('next_ts',F.lag('ts',1).over(window_user))\
        .withColumn('next_action',F.lag('page',1).over(window_user))\
        .withColumn("diff_ts",(F.col('next_ts').cast('integer')- F.col('ts').cast('integer'))/1000)

    # keep only the Nextsong action , filter 
    df_listen_session_song = df_listen_session.filter(F.col('page')=='NextSong')\
       .withColumn("listen_session",F.sum("diff_ts").over(window_session))
    
    # extract max value only for each session per user
    df_listen_session_song = df_listen_session_song.groupby(['userId','level','week_code','sessionId'])\
        .agg(F.max('listen_session').alias('total_listen_session'),\
            F.max('itemInSession').alias('item_session'))
    
    df_listen_session_song = df_listen_session_song.withColumn('avg_listen_item_session',
        F.round((F.col('total_listen_session')/F.col('item_session'))/60,2))       
    
    
    # add a column with total number of session , avg_listen_time per session
    num_session = df_week.groupby(['userId','level','week_code'])\
        .agg(F.countDistinct('sessionId').alias('num_session'))
    
    # join 
    df_listen_session_song = df_listen_session_song.join(num_session, on=['userId','level','week_code'], how='left')
    
    df_listen_session_song = df_listen_session_song.withColumn("week_total_listen",
                            F.sum('total_listen_session').over(Window.partitionBy('userId','week_code')))\
        .withColumn('week_avg_listen_item_session',F.round(F.sum('avg_listen_item_session').over(Window.partitionBy('userId','week_code'))/F.col('num_session'),2))\
        .withColumn('week_avg_listen_time_total_session',F.round((F.col('week_total_listen')/F.col('num_session'))/60,2))\
        
    
    # final dataframe
    df_listen = df_listen_session_song.select(['userId','level','week_code','num_session','week_avg_listen_item_session','week_avg_listen_time_total_session'])\
    .distinct()\
    .withColumn('listen_code',F.monotonically_increasing_id()+1)
    
    df_listen.write.parquet(output_path+"/dim_listen/{}_{}".format(start_date, end_date),mode="overwrite")
    
    return df_listen

def processing_recent(df_week,output_path,start_date, end_date):
    """
        유저의 최근 활동기간 및 가입기간으로부터 현재까지의 누적날짜 집계 후 
        테이블 생성
    """
    df_last_time = df_week.groupby(['userId','level'])\
    .agg(F.max('date').alias('last_interaction'))

    df_regi = df_week.select(['userId','level','regi_date']).distinct()
    
    df_recent = df_last_time.join(df_regi,on=['userId','level'],how='left')\
        .withColumn('day_from_reg',F.datediff('last_interaction','regi_date'))\
        .drop('regi_date')\
        .withColumn('recent_code',F.monotonically_increasing_id()+1)

    df_recent.write.parquet(output_path+"/dim_recent/{}_{}".format(start_date, end_date),mode="overwrite")
    
    return df_recent

def processing_action(df_week,output_path,start_date, end_date):
    """
        유저의 주간별 세션 내에서의 활동 횟수 및 세션별 평균 활동 횟수 집계 
        및 테이블 생성
    """
    
    window_user = Window.partitionBy("userId").orderBy(F.desc('ts'))
    window_session = Window.partitionBy(['userId','sessionId']).orderBy('ts').rangeBetween(Window.unboundedPreceding,0)
    
    
    df_action_session = df_week.groupby(['userId','level','week_code','sessionId'])\
            .agg(F.count('page').alias('action_per_session'))
    df_action_session = df_action_session.withColumn('total_action_week',\
                                    F.sum('action_per_session').over(Window.partitionBy('userId','week_code')))
    
    num_session = df_week.groupby(['userId','level','week_code'])\
            .agg(F.countDistinct('sessionId').alias('num_session'))
    
    df_action_merge = df_action_session.join(num_session,on=['userId','level','week_code'],how='left')\
    .withColumn('avg_num_action_session',F.round(F.col('total_action_week')/F.col('num_session'),2))
    
    df_action_merge = df_action_merge.select(['userId','level','week_code','total_action_week','num_session','avg_num_action_session'])\
    .distinct()\
    .withColumn('action_code',F.monotonically_increasing_id()+1)
    
    df_action_merge.write.parquet(output_path+"/dim_action/{}_{}".format(start_date,end_date),mode="overwrite")
    
    return df_action_merge

def get_users(df_week):
    """
        Extract the list of unique combination of userId, level
    """
    service_users = df_week.select(['userId','level','week_code']).distinct()
    return service_users

def load_dim_table(spark_session,output_path,dim_table_name,start_date, end_date):
    """
        Load saved dimenstion table as format parquet
    """
    
    return spark_session.read.parquet(output_path+dim_table_name+"/{}_{}".format(start_date,end_date))

def processing_fact_table(spark,df_week,output_path,start_date, end_date):
    
    # load dim_table 
    df_user = get_users(df_week)
    df_dim_churn = load_dim_table(spark,output_path,"dim_churn_label",start_date, end_date)
    df_dim_recent = load_dim_table(spark,output_path,"dim_recent",start_date, end_date)
    df_dim_login = load_dim_table(spark,output_path,"dim_login",start_date, end_date)
    df_dim_song_ads = load_dim_table(spark,output_path,"dim_song_ads",start_date, end_date)
    df_dim_listen = load_dim_table(spark,output_path,"dim_listen",start_date, end_date)
    df_dim_repeat = load_dim_table(spark,output_path,"dim_repeat",start_date, end_date)
    df_dim_action = load_dim_table(spark,output_path,"dim_action",start_date, end_date)
    
    # set columns to warehouse dataframe
    warehouse_cols = ['userId','level','last_interaction','week_code','day_from_reg','churn_service','churn_paid','count_login','time_inter_login'\
                     ,'count_song','count_song_weekday','count_song_weekend','count_ads','ratio_song_ads','count_repeat','repeat_ratio','week_avg_listen_item_session','total_action_week',\
                 'avg_num_action_session']
    
    df_fact_table = df_user.join(df_dim_churn, on=['userId'],how='left')\
        .join(df_dim_recent,on=['userId','level'],how='inner')\
        .join(df_dim_login.select(['userId','week_code','count_login','time_inter_login']),on=['userId','week_code'],how='inner')\
        .join(df_dim_song_ads, on=['userId','level','week_code'],how='inner')\
        .join(df_dim_listen, on=['userId','level','week_code'],how='inner')\
        .join(df_dim_repeat,on=['userId','level','week_code'],how='inner')\
        .join(df_dim_action, on=['userId','level','week_code'],how='inner')\
        .select(warehouse_cols)

    


    # save to warehouse
    df_fact_table.write.parquet(output_path+"/user_week_summary/"+"{}_{}".format(start_date,end_date), mode='overwrite')
    