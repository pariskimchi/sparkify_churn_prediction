import configparser
import os
import logging
from tracemalloc import start

import findspark
findspark.init()
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
from pyspark.sql.types import StructField, StructType, IntegerType, FloatType, StringType 

# modeling 
from pyspark.sql.functions import split 
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier 
from pyspark.ml.evaluation  import MulticlassClassificationEvaluator 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from util import convert_registration, convert_ts, drop_na, extract_location_state, load_data_format, set_churn_label
from utility import WeekSummary,UserSummary,cleaned_df

# setup logging 

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["PATH"] = "/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/jvm/java-8-openjdk-amd64/bin"
# os.environ["SPARK_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"
# os.environ["HADOOP_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"

# os.environ["JAVA_HOME"] = "/usr/bin/java"

# AWS configuration 
config = configparser.ConfigParser()
config.read('config.cfg',encoding='utf-8-sig')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']
SOURCE_S3_BUCKET = config['S3']['SOURCE_S3_BUCKET']
DEST_S3_BUCKET = config['S3']['DEST_S3_BUCKET']

# SOURCE_REDSHIFT = config['REDSHIFT']['SOURCE_REDSHIFT']
# DEST_REDSHIFT = config['REDSHIFt']['DEST_REDSHIFT']


origin_path = ""
dest_path = ""


AWS_ACCESS_KEY_ID     = config.get('AWS', 'AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('AWS', 'AWS_SECRET_ACCESS_KEY')

data_path = "./mini_sparkify_event_data.json"

# data processing functions
def create_spark_session():
    """
        Create spark session object
    
    """
    # os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    # os.environ["PATH"] = "/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/jvm/java-8-openjdk-amd64/bin"
    # os.environ["SPARK_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"
    # os.environ["HADOOP_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"

    spark = SparkSession \
        .builder \
        .config("spark.jars.packages","saurfang:spark-sas7bdat:2.0.0-s_2.11,org.apache.hadoop:hadoop-aws:2.7.2") \
        .config("fs.s3a.access.key", AWS_ACCESS_KEY_ID) \
        .config("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY) \
        .config("spark.network.timeout","10000s")\
        .config("spark.executor.heartbeatInterval","3600s")\
        .config("spark.executor.memory","100g")\
        .config("spark.shuffle.file.buffer", "64k")\
        .config("spark.eventLog.buffer.kb", "200k")\
        .config("spark.sql.broadcastTimeout", 7200)\
        .getOrCreate()


    return spark


# Preprocessing 
# From lake => df_clean 

def process_clean_df(spark,SOURCE_S3_BUCKET, DEST_S3_BUCKET):

    # logging.info("Starting process cleaning origin dataframe")

    # load raw dataset from s3 using `origin_path`
    ## df = spark.read.format("")
    # df = load_data_format(spark)
    df = spark.read.json(path=SOURCE_S3_BUCKET+data_path)

    df_clean = cleaned_df(df)

    # save cleaned dataframe on redshift? or S3
    # df_clean.write.parquet(DEST_S3_BUCKET+"lake_clean_df",mode="overwrite")

    df_clean.write.parquet(DEST_S3_BUCKET+"lake_clean_df",mode="overwrite")




    

def load_clean_df(spark,clean_df_path):
    """
        Load cleaned dataframe from s3 data lake where cleaned dataframe is stored
    """

    df_clean = spark.read.parquet(clean_df_path)
    
    return df_clean


# processing get_week_df
def get_week_from_df(df, week_start_date,week_end_date):
    """
        Function to extract the rows corresponding to a given week
    """

    # filter by week_start_date, week_end_date
    filter_week = (F.col('date')>=week_start_date) &(F.col('date')<=week_end_date)

    # # is_weekday or not 
    # is_weekday = udf(lambda x:1 if x in ['1','2','3','4','5'] else 0, IntegerType())
    # is_ads = udf(lambda x:1 if x == "Roll Advert" else 0, IntegerType())

    def weekday_weekend():
        col = F.when(F.col('dayofweek').isin(['1','2','3','4','5']),F.lit(1))\
                .otherwise(F.lit(0))
        
        return col

    def is_ads():
        col = F.when(F.col('page')=='Roll Advert',F.lit(1))\
                .otherwise(F.lit(0))

        return col

    # apply filter_week 
    df_week = df.filter(filter_week).orderBy('ts',ascending=False)\
        .withColumn('is_weekday',weekday_weekend())\
        .withColumn('is_ads',is_ads())

    return df_week

# Processing weekSummary 
def processing_week_summary(spark, start_date, end_date, input_path, output_path):
    """
        Processing to create a summary dataframe on a given time series

        - last interaction date 
        - count of songs(total, on weekdays only, on weekends only), distinct artists,
            ads, repeats, logins
        - avg delta time between login 
        - avg listening time per session 
        - avg number of actions per session

    """

    logger.info(" Start Processing user week summary {}~{}".format(start_date, end_date))

    df = spark.read.parquet(DEST_S3_BUCKET+"lake_clean_df")
    # get week df 
    df_week = get_week_from_df(df,start_date,end_date)

    # load week_summary 
    week_sum = WeekSummary(df_week)
    week_sum_df = week_sum.week_summary
    logger.info("completed reading week summary dataframe on {}~{}".format(start_date, end_date))

    # save on redshift 
    logger.info("Start writing week_summary parquet!!!!")
    # week_sum_df.write.mode("overwrite")\
    #         .partitionBy("userId")\
    #         .parquet(path=output_path+"week_sum_{}_{}".format(start_date,end_date))
    week_sum_df.write.parquet(output_path+"week_sum_{}_{}".format(start_date,end_date),mode='overwrite')



    logger.info("Success Processing user week summary {}~{}".format(start_date, end_date))

    #




# Processing UserSummary

def processing_user_summary(spark,table_name,week_table_name, input_path, output_path):
    """
        Processing user summary dataframe partition by user_id
        from s3 to data warehouse path 

    """
    
    df_clean = spark.read.parquet(DEST_S3_BUCKET+"lake_clean_df")
    # create empty User summary dataframe
    user_summary_week = UserSummary(spark,df_clean,table_name)

    # loading saved week_summary 
    week_summary = spark.read.parquet(output_path+week_table_name)
    
    # update user summary from given week summary
    user_summary_week.update_user_summary(week_summary)

    user_summary_week_df = user_summary_week.load_summary()
    # save user_summary on warehouse 
    user_summary_week_df.write.parquet(output_path+table_name)


def main():

    logging.info("SparkSession Created!")
    spark = create_spark_session()

    input_data = SOURCE_S3_BUCKET
    output_data = DEST_S3_BUCKET

    logging.info("Process cleaning started!")
    # upload cleaned dataframe to s3 
    process_clean_df(spark,input_data, output_data)
    logging.info("Data Cleaning completed!")


    start_date = "2018-10-01"
    end_date = "2018-10-07"
    logging.info("Start Date: {}".format(start_date))
    logging.info("End Date:{}".format(end_date))

    # upload week summary parquet to s3
    processing_week_summary(spark,start_date,end_date,input_data,output_data)

    week_table_name="week_sum_{}_{}".format(start_date,end_date)
    table_name = "sparkify_user_summary"

    # upload user summary parquet to s3
    processing_user_summary(spark,table_name, week_table_name,input_data, output_data)


    logging.info("Data Processing completed!")

if __name__ == "__main__":
    main()
