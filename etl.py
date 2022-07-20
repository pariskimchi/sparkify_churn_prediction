import configparser
import os
import logging
from tracemalloc import start

import findspark
findspark.init()
import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt 

import pyspark 
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
from pyspark.sql.functions import udf, isnan, col, upper 
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

from utility2 import  processing_calendar_week,processing_churn_label,processing_login,get_users,processing_song_ads,processing_recent,processing_action,processing_listen,processing_repeat,processing_fact_table
from utility2 import load_dim_table, load_data_format, load_clean_df
from utility2 import get_week_from_df,processing_cleaning_df,save_cleaned_df
# setup logging 

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS configuration 
config = configparser.ConfigParser()
config.read('config.cfg',encoding='utf-8-sig')
SOURCE_S3_BUCKET = config.get('S3', 'SOURCE_S3_BUCKET')
DEST_S3_BUCKET= config.get('S3', 'DEST_S3_BUCKET')

AWS_ACCESS_KEY_ID= config.get('AWS', 'AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('AWS', 'AWS_SECRET_ACCESS_KEY')

data_path = "mini_sparkify_event_data.json"
# data_path = "./sparkify_event_data.json"

# full_data_path = "s3n://udacity-dsnd/sparkify/sparkify_event_data.json"
# data processing functions
def create_spark_session():
    """
        Create spark session object
    
    """
    spark = SparkSession \
        .builder \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.2')\
        .config("fs.s3a.access.key", AWS_ACCESS_KEY_ID) \
        .config("fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY) \
        .config("spark.network.timeout","10000s")\
        .config("spark.executor.heartbeatInterval","3600s")\
        .config("spark.driver.memory","40g")\
        .config("spark.executor.memory","120g")\
        .config("spark.shuffle.file.buffer", "64k")\
        .config("spark.eventLog.buffer.kb", "200k")\
        .config("spark.sql.broadcastTimeout", 7200)\
        .config("spark.master","local[12]")\
        .getOrCreate()


    return spark


# Preprocessing 
# From lake => df_clean 

def process_clean_df_full(spark,data_path, output_path):
    """
        Data Cleaning on raw dataset
    """
    # logging.info("Starting process cleaning origin dataframe")

    # load raw dataset 
    df = spark.read.json(path=SOURCE_S3_BUCKET+data_path)

    # First, Run Processing cleaning funciton 
    processing_cleaning_df(spark,df,output_path)
    
    # Second, Save merged parquet as df_clean 
    save_cleaned_df(spark,output_path)
  


def load_clean_df(spark,clean_df_path):
    """
        Load cleaned dataframe from s3 data lake where cleaned raw dataset is stored
    """

    df_clean = spark.read.parquet(clean_df_path)
    
    return df_clean


# build up dimension function


def main():

    logging.info("SparkSession Created!")
    spark = create_spark_session()

    input_path = SOURCE_S3_BUCKET
    output_path = DEST_S3_BUCKET

    logging.info("Process cleaning started!")
    # upload cleaned dataframe to s3 
    process_clean_df_full(spark,data_path, output_path)
    logging.info("Complete processing Cleaning dataframe ")


    clean_df_path = output_path+"/lake_cleaned_df_full"
    # Third, Load df_clean using s3_path
    df_clean = load_clean_df(spark,clean_df_path)
    logger.info("Completed Loading clean_df from s3")

    start_date = "2018-10-01"
    end_date = "2018-10-21"
    logging.info("Start Date: {}".format(start_date))
    logging.info("End Date:{}".format(end_date))


    # ### build up dimension function as one function
    # # processing dimension tables 
    processing_calendar_week(df_clean,output_path)
    df_week = get_week_from_df(spark,df_clean,output_path,start_date,end_date)

    processing_churn_label(df_week,output_path,start_date,end_date)

    processing_login(df_week,output_path,start_date, end_date)

    processing_song_ads(df_week,output_path,start_date, end_date)

    processing_repeat(df_week,output_path,start_date, end_date)

    processing_listen(df_week,output_path,start_date, end_date)

    processing_recent(df_week,output_path,start_date, end_date)

    processing_action(df_week,output_path,start_date, end_date)

    # ####### processing fact_table
    processing_fact_table(spark,df_week,output_path,start_date, end_date)
    logging.info("Data Processing completed!")

if __name__ == "__main__":
    main()