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

origin_path = ""
dest_path = ""


AWS_ACCESS_KEY_ID     = config.get('AWS', 'AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('AWS', 'AWS_SECRET_ACCESS_KEY')

# data_path = "./mini_sparkify_event_data.json"
data_path = "./sparkify_event_data.json"

# data processing functions
def create_spark_session():
    """
        Create spark session object
    
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages","saurfang:spark-sas7bdat:2.0.0-s_2.11,org.apache.hadoop:hadoop-aws:2.7.2") \
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
    logging.info("Start Loading Raw data From {}".format(data_path))
    df = spark.read.json(path=data_path)
    logging.info("Complete Loading Raw data")

    # First, Run Processing cleaning funciton 
    logger.info("Start processing_cleaning_df")
    processing_cleaning_df(spark,df,output_path)
    logger.info("Completed processing cleaning df")

    # Second, Save merged parquet as df_clean 
    logger.info("Start Saved_cleaned_df")
    save_cleaned_df(spark,output_path)
    logger.info("Completed Save cleaned_df")


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
    logging.info("Processing Creating Calendar week!")
    processing_calendar_week(df_clean,output_path)
    logging.info("Complete Creating Calendar week!")
    df_week = get_week_from_df(spark,df_clean,output_path,start_date,end_date)

    logging.info("Processing Creating Churn label week!")
    processing_churn_label(df_week,output_path,start_date,end_date)
    logging.info("Complete Creating churn label week!")

    logging.info("Processing Creating login week!")
    processing_login(df_week,output_path,start_date, end_date)
    logging.info("Complete Creating login week!")

    logging.info("Processing Creating song_ads week!")
    processing_song_ads(df_week,output_path,start_date, end_date)
    logging.info("Complete Creating song_ads week!")

    logging.info("Processing Creating repeat week!")
    processing_repeat(df_week,output_path,start_date, end_date)
    logging.info("Complete Creating repeat week!")

    logging.info("Processing Creating listen week!")
    processing_listen(df_week,output_path,start_date, end_date)
    logging.info("Complete Creating listen week!")


    logging.info("Processing Creating recent week!")
    processing_recent(df_week,output_path,start_date, end_date)
    logging.info("Complete Creating recent week!")

    logging.info("Processing Creating action week!")
    processing_action(df_week,output_path,start_date, end_date)
    logging.info("Complete Creating action week!")


    # ####### processing fact_table
    processing_fact_table(spark,df_week,output_path,start_date, end_date)
    logging.info("Data Processing completed!")

if __name__ == "__main__":
    main()