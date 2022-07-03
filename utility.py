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
def drop_na(df):
    """
        drop null value on userId, sessionId
    """
    
    # define filter condition 
    filter_user_id = (df['userId'] != "") & (df['userId'].isNotNull())& (~isnan(df['userId']))
    filter_session_id = (df['sessionId'].isNotNull()) & (~isnan(df['sessionId']))
    
    df_clean = df.filter(filter_user_id).filter(filter_session_id)
    
    return df_clean

# convert ts to timestamp, an dcreate date, year, month, day, hour, weekday, weekofyear
def convert_ts(df):
    """
        Convert timestamp column into serveral time columns 
    """
    ts = (F.col('ts')/1000).cast('timestamp')
    
    df_clean = df.withColumn('date',F.date_format(ts,format='yyyy-MM-dd'))\
        .withColumn('date',F.to_date(F.col('date'),'yyyy-MM-dd'))\
        .withColumn('year',F.year(F.col('date')))\
        .withColumn('month',F.month(F.col('date')))\
        .withColumn('day',F.dayofmonth(F.col('date')))\
        .withColumn('hour', F.hour(ts))\
        .withColumn('dayofweek',F.dayofweek(F.col('date')))\
        .withColumn('weekofyear',F.weekofyear(F.col('date')))
    
    return df_clean

# convert ts to timestamp, an dcreate date, year, month, day, hour, weekday, weekofyear
def convert_registration(df):
    """
        Convert registration column into serveral time columns 
    """
    regi_ts = (F.col('registration')/1000).cast('timestamp')

    df_regi = df.withColumn('regi_date',F.date_format(regi_ts,format='yyyy-MM-dd'))\
        .withColumn('regi_date',F.to_date(F.col('regi_date'),'yyyy-MM-dd'))\
            .withColumn('regi_year',F.year(F.col('regi_date')))\
            .withColumn('regi_month',F.month(F.col('regi_date')))\
            .withColumn('regi_day',F.dayofmonth(F.col('regi_date')))\
            .withColumn('regi_hour', F.hour(regi_ts))\
            .withColumn('regi_dayofweek',F.dayofweek(F.col('regi_date')))\
            .withColumn('regi_weekofyear',F.weekofyear(F.col('regi_date')))
    
    return df_regi
    
# extract location_state
def extract_location_state(df):
    """
        splits the location column to extract state
    """
    df_extract =  df.withColumn('state',F.split(F.col('location'),", ").getItem(1))
    
    return df_extract

def set_churn_label(df):
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
    # # udf to flag churn_service, churn_paid 
    # add_cancelled = udf(lambda x:1 if x == "Cancellation Confirmation" else 0, IntegerType())
    # add_downgraded = udf(lambda x:1 if x== "Submit Downgrade" else 0, IntegerType())
    # add_upgraded = udf(lambda x:1 if x=="Submit Upgrade" else 0, IntegerType())
    
    # apply udf and create flag column 
    df_flag = df.withColumn('cancelled',add_cancelled())\
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
    
    return df_churn


# # prepared df
def cleaned_df(df):
    """
        clean and format dataframe 
    """
    df_clean = drop_na(df)
    df_clean = convert_ts(df_clean)
    df_clean = convert_registration(df_clean)
    df_clean = extract_location_state(df_clean)
    df_clean = set_churn_label(df_clean)
    
    return df_clean


def load_clean_df(spark,clean_df_path):
    """
        Load cleaned dataframe from s3 data lake where cleaned dataframe is stored
    """

    df_clean = spark.read.parquet(clean_df_path)
    
    return df_clean


def get_week_from_df(df, week_day_start, week_day_last):
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
    filter_week = (F.col('date')>=week_day_start) &(F.col('date')<=week_day_last)
    
    

    # # is_weekday or not 
    # is_weekday = udf(lambda x:1 if x in [1,2,3,4,5] else 0, IntegerType())
    # is_ads = udf(lambda x:1 if x == "Roll Advert" else 0, IntegerType())
    
    # apply filter_week 
    df_week = df.filter(filter_week).orderBy(F.desc('ts'))\
        .withColumn('is_weekday',weekday_weekend)\
        .withColumn('is_ads',is_ads())
    return df_week



class WeekSummary:
    """
    
    """
    
    def __init__(self, df_week):
        # get df from given weeek
        self.df_week = df_week
        # get unique userId ,level
        self.users = self.get_users()
        # get week summary 
        self.week_summary = self.get_week_summary()
    
    def get_users(self):
        """
            Extract the list of unique combination of userId, level
        """
        service_users = self.df_week.select(['userId','level']).distinct()
        return service_users
    
    #last_interaction
    def get_last_interaction(self):
        """
            get last interaction date from user
        """
        df_last_time = self.df_week.groupby(['userId','level'])\
            .agg(F.max('date').alias('last_interaction'))
            
        
        return df_last_time
    
    
    def compute_count_song(self):
        """
            extract the count of songs per user during week or weekend 
            1. total count of songs 
            2. total song per weekdays 
            3. total song per weekend
        """
        song_week = self.df_week.groupby(['userId','level'])\
            .agg(F.count('song').alias('count_song'))
        
        song_weekday = self.df_week.filter(F.col('is_weekday')==1)\
            .groupby(['userId','level'])\
            .agg(F.countDistinct('song').alias('count_song_weekday'))
    
        song_weekend = self.df_week.filter(F.col('is_weekday')!=1)\
            .groupby(['userId','level'])\
            .agg(F.countDistinct('song').alias('count_song_weekend'))
        
        return song_week, song_weekday, song_weekend
    
    def compute_count_artist(self):
        """
            extract the count of distinct artist listend to per user
        """
        artist_week = self.df_week.groupby(['userId','level'])\
            .agg(F.countDistinct('artist').alias('count_distinct_artist'))
        
        return artist_week
    
    def compute_count_login(self):
        """
            Extract the number of session per user
        """
        login_week = self.df_week.groupby(['userId','level'])\
            .agg(F.countDistinct('sessionId').alias('count_login'))
        
        
        return login_week
    
    def compute_count_ads(self):
        """
            Extract the number of ads listened to per user
        """
        ads_week = self.df_week.groupby(['userId','level'])\
            .agg(F.sum('is_ads').alias('count_ads'))
        
        return ads_week
    
    #get_last_interaction
    #compute_counts
    def compute_count_merge(self):
        """
            add column per count
        """
        df_count = self.users
        
        # add songs on weekdays and weekend 
        song_week, song_weekday, song_weekend = self.compute_count_song()
        
        df_count = df_count.join(song_week, on=['userId','level'], how='full')\
            .join(song_weekday, on=['userId','level'], how='full')\
            .join(song_weekend, on=['userId','level'], how='full')
        
        # add distinct artists 
        artist_week = self.compute_count_artist()
        df_count = df_count.join(artist_week, on=['userId','level'], how='full')
        
        # add login (unique sessionId?)
        login_week = self.compute_count_login()
        df_count = df_count.join(login_week, on=['userId','level'], how='full')
        
        # add ads
        ads_week = self.compute_count_ads()
        df_count = df_count.join(ads_week, on=['userId','level'], how='full')
        
        # repeat
        df_count = df_count.withColumn('count_repeat',F.col('count_song')-(F.col('count_song_weekday')+F.col('count_song_weekend')))
        
        return df_count
        
    def compute_delta_login(self):
        """
            Calculate avg delta time between two login in the given week
        """
        
        # set inwdow 
        window_login = Window.partitionBy(['userId']).orderBy(F.desc('ts'))
        
        # add new column next date 
        df_delta = self.df_week.withColumn('next_date',F.lag('date',1).over(window_login))
        
        # calculate delta betwwen two login
        df_delta = df_delta.withColumn('delta_time',F.datediff(F.col('next_date'),F.col('date')))
        
        # compute the avg delta_time between login for each user 

        delta_cols = ['userId','level','churn_service','churn_paid','delta_time']

        df_delta_login = df_delta.select(delta_cols)\
            .filter(df_delta['delta_time'] != 0)
        distinct_session = self.df_week.groupby('userId').agg(F.countDistinct('sessionId').alias('user_num_session'))

        # join: df_delta_login + distinct_session
        df_delta_login = df_delta_login.join(distinct_session, on=['userId'], how='inner')

        # calculate avg delta time , total_avg_delta
        df_delta_login = df_delta_login.withColumn('avg_delta',F.col('delta_time')/F.col('user_num_session'))\
            .withColumn('total_avg_delta',F.sum('avg_delta').over(Window.partitionBy('userId')))
        
        # group by , relevant data 
        df_delta_login = df_delta_login.groupby(['userId','level'])\
            .agg(F.max('total_avg_delta').alias('time_inter_login'))
        
        
        return df_delta_login
    
    def compute_listen_session(self):
        """
            Calculate the average listening per session per user 
        """
        # window per user by desc timestmp
        window_user = Window.partitionBy("userId").orderBy(F.desc('ts'))
        #window per user, session
        window_session = Window.partitionBy(["userId","sessionId"]).orderBy("ts").rangeBetween(Window.unboundedPreceding,0)
        
        # add two new columns: next_ts, next_action
        df_listen_session = self.df_week.withColumn('next_ts',F.lag('ts',1).over(window_user))\
            .withColumn('next_action',F.lag('page',1).over(window_user))
        
        # calculate the diff between two timestamp
        df_listen_session = df_listen_session.withColumn("diff_ts",(F.col('next_ts').cast('integer')- F.col('ts').cast('integer'))/1000)
        
        # keep only the Nextsong action , filter 
        df_listen_session_song = df_listen_session.filter(F.col('page')=='NextSong')
        # add a column total listening 
        df_listen_session_song = df_listen_session_song.withColumn("listen_session",F.sum("diff_ts").over(window_session))
        
        # extract max value only for each session per user
        df_listen_session_song = df_listen_session_song.groupby(['userId','sessionId','level'])\
            .agg(F.max('listen_session').alias('total_listen_session'),\
                F.max('itemInSession').alias('item_session'))
        
        df_listen_session_song = df_listen_session_song.withColumn('avg_listen_session',
            F.round((F.col('total_listen_session')/F.col('item_session'))/60,2))                                            
        
        # add a column with total number of session , avg_listen_time per session
        num_session = self.df_week.groupby(['userId','level'])\
            .agg(F.countDistinct('sessionId').alias('num_session'))
        
        df_listen_session_song = df_listen_session_song.join(num_session, on=['userId','level'], how='full')
        
        df_listen_session_song = df_listen_session_song.withColumn("week_total_listen",
                            F.sum('avg_listen_session').over(Window.partitionBy('userId')))\
            .withColumn('avg_listen_time_session',F.round((F.col('week_total_listen')/F.col('num_session')),2))
        
        # keep relevant columns and distinct 
        df_listen_session_song = df_listen_session_song.select(['userId','level','avg_listen_time_session']).distinct()
    
        return df_listen_session_song
        
        
    def compute_action_session(self):
        """
            calculate the avg number of action per session per user for the given week
        """
        
        window_user = Window.partitionBy("userId").orderBy(F.desc('ts'))
        window_session = Window.partitionBy(['userId','sessionId']).orderBy('ts').rangeBetween(Window.unboundedPreceding,0)
        
        #
        df_action_session = self.df_week.groupby(['userId','level','sessionId'])\
            .agg(F.count('page').alias('action_per_session'))
        df_action_session = df_action_session.withColumn('total_action_week',\
                                        F.sum('action_per_session').over(Window.partitionBy('userId')))
        
        # add column with total number of session in the week 
        
        num_session = self.df_week.groupby(['userId','level'])\
            .agg(F.countDistinct('sessionId').alias('num_session'))
        
        df_action_session = df_action_session.join(num_session, on=['userId','level'], how='full')
        
        df_action_session = df_action_session.withColumn('avg_num_action_session',F.round(F.col('total_action_week')/F.col('num_session'),2))
        
        # keep only the relevant columns 
        df_action_session = df_action_session.select(['userId','level','avg_num_action_session']).distinct()
        
        return df_action_session
    
    
    def compute_avg(self):
        """
            Add a columns per avg to avg_df one row per user for the given week
        """
        
        # only specific user dataframe
        df_user_avg = self.users
        
        # delta time between login
        df_delta_login = self.compute_delta_login()
        df_user_avg = df_user_avg.join(df_delta_login, on=['userId','level'], how='full')
        
        # time per session
        df_listen_session = self.compute_listen_session()
        df_user_avg = df_user_avg.join(df_listen_session, on=['userId','level'], how='full')
        
        # action per session
        df_action_session = self.compute_action_session()
        df_user_avg = df_user_avg.join(df_action_session, on=['userId','level'], how='full')
        
        return df_user_avg
    
    #replace_na
    def replace_na(self, df):
        """
            Sets default values for null values when calculating count and average
        """
        # replace null value of time_inter_login to 7 (number of days 7)
        df_filled = df.na.fill({'time_inter_login':'7'})
        df_filled = df_filled.na.fill(0)
        
        return df_filled
    
    
    
    def get_week_summary(self):
        """
            여기서 최종적으로 week_summary준다 
            1. last_interaction time
            2. count(song, ads,login, artist distcint, song repeat)
            3. avg()
            4. join(df_avg,on=['userId','level'])
            5. after join, replace nan value
        """
        # 최근 활동시간 
        logging.info("start get_last_interaction")
        week_last_interaction = self.get_last_interaction()
        logging.info("finished get_last_interaction")

        # 여러가지 count 
        logging.info("start comopute_count_merge")
        week_summary_count = self.compute_count_merge()
        logging.info("finished compute_count_merge")

        # join
        logging.info("start last + compute_count join")
        week_summary = week_last_interaction.join(week_summary_count, on=['userId','level'], how='full')
        logging.info("finished last + compute_count join")

        # avg 
        logging.info("Start compute _avg")
        week_summary_avg = self.compute_avg()
        logging.info("finished compute avg")

        # summary.join(avg)
        logging.info("start week_summary + join + compute_avg")
        week_summary = week_summary.join(week_summary_avg, on=['userId','level'], how='full')
        logging.info("finished week_summary + join + compute_avg")

        # replace nan value 
        logging.info("start replace_NAN value")
        week_summary = self.replace_na(week_summary)
        logging.info("Finished replace NAN value")
        
        return week_summary



class UserSummary:
    """
    
    """
    
    def __init__(self,spark,df, table_name):
        self.spark = spark
        self.df = df 
        self.table_name = table_name
        self.user_summary = None
    
    def create_column_with_type(self,df,column_list, target_type):
        """

        """
        for column in column_list:
            df = df.withColumn(column,F.lit(None))\
                .withColumn(column, F.col(column).cast(target_type))

        return df
    
    def load_summary(self):
        """
            Load existing summary 
        """
        
        try:
            loaded_summary = self.spark.read.parquet('./spark-warehouse/{}'.format(self.table_name))
            return loaded_summary
        
        except:
            # create an empty dataframe 
            # 1. using StructField 
            user_summary = self.spark.createDataFrame([],StructType([]))
            
            str_cols = ['userId','level','gender','state']
            int_cols = ['churn_service','churn_paid','count_song','count_song_weekday','count_song_weekend','count_distinct_artist','count_login','count_ads','count_repeat','day_from_reg']
            float_cols = ['time_inter_login','avg_listen_time_session','avg_num_action_session']
            date_cols = ['last_interaction']
            # apply create_column_type
            user_summary = self.create_column_with_type(user_summary,str_cols,StringType())
            user_summary = self.create_column_with_type(user_summary,int_cols,IntegerType())
            user_summary = self.create_column_with_type(user_summary,float_cols,FloatType())
            user_summary = self.create_column_with_type(user_summary,date_cols,DateType())
            # 2. columns = []
            # schema= StructType(columns)
            #. user_summary = spark.createDataFrame(data=[],schema=schema)
            # convert_type, integeer_cols, string_cols, float_cols
            
            
            return user_summary
    
    def save_summary(self,mode):
        """
            Save summary as data warehouse
        """
        self.user_summary.write\
            .mode(mode)\
            .parquet(self.table_name)
            
        
    
    # update 1.day_from_reg
    def compute_day_from_reg(self, updated_summary):
        """
            Computes day from reg using last_interaction
        """
        regi_summary = self.df.select(['userId','level','regi_date']).distinct()
        
        #join input_df + reg_summary
        updated_summary = updated_summary.join(regi_summary,on=['userId','level'])
        
        # calculate day_from_reg, with F.datediff().
        updated_summary = updated_summary.withColumn('day_from_reg',\
                                                    F.datediff('last_interaction','regi_date'))
        
        # drop regi_date
        updated_summary = updated_summary.drop('regi_date')
        
        return updated_summary 
    
    
    # init_user_summary 
    def init_user_summary(self, updated_summary):
        """
        
        """
        updated_summary = self.compute_day_from_reg(updated_summary)
        
        return updated_summary
    
    def get_last_summary(self):
        """
            return last version of the summary(version saved) and rename the columns
        """
        # rename으로 구분
        
        user_summary = self.user_summary.createOrReplaceTempView("user_summary")
        user_summary = self.spark.sql("SELECT * FROM user_summary")
        
        last_user_summary = self.spark.sql("""
            SELECT * FROM (
                SELECT *, MAX(last_interaction) OVER (PARTITION BY userId,level) AS max_last FROM user_summary
            ) table1 \
            WHERE last_interaction = table1.max_last
        """)
        
        # rename count_cols => last_count_cols 
        count_cols =['count_song','count_song_weekday','count_song_weekend',\
                    'count_distinct_artist','count_login','count_ads','count_repeat',\
                    'time_inter_login','avg_listen_time_session','avg_num_action_session']
        # for loop
        # apply rename column
        for col in count_cols:
            last_user_summary = last_user_summary.withColumnRenamed(col, 'last_{}'.format(col))
            
        return last_user_summary
    
    # update old summary 
    def update_old_summary(self, updated_summary):
        """
            Appends new rows for each user for the next week, to prev loaded summary.
            
        """
        last_user_summary = self.get_last_summary()
        
        # drop last cols
        drop_old_cols = ['last_interaction','gender','state','churn_service','churn_paid','max_last']
        
        for col in drop_old_cols:
            last_user_summary = last_user_summary.drop(col)
#         last_user_summary = last_user_summary.drop(drop_old_cols)
        
        # split the user_summary into 2df: one with knownuser, one with new user 
        # for new user, apply instantiation of the summary 
        new_user_summary = updated_summary.join(last_user_summary, on=['userId','level'],\
                                               how='left_anti')
        
        new_user_summary = new_user_summary.select(['userId','level','last_interaction',\
                                                   'count_song','count_song_weekday','count_song_weekend',\
                                                   'count_distinct_artist','count_login','count_ads','count_repeat',\
                                                   'time_inter_login','avg_listen_time_session','avg_num_action_session',\
                                                   'gender','state','churn_service','churn_paid'])
        
        # create new user summary  with calculatting day_from_reg
        new_user_summary = self.init_user_summary(new_user_summary)
        
        # for old user too, day_from_reg
        old_user_summary = updated_summary.join(last_user_summary, on=['userId','level'],how='inner')
        
#         print(old_user_summary.columns)
        # drop column last_{} count column 
        count_cols = ['count_song', 'count_song_weekday', 'count_song_weekend',\
                     'count_distinct_artist', 'count_login', 'count_ads', 'count_repeat',\
                     'time_inter_login', 'avg_listen_time_session', 'avg_num_action_session']
        
        for column in count_cols:
            last_count_col='last_{}'.format(column)
            old_user_summary = old_user_summary.drop(last_count_col)
    
        
        # finally add day_from_reg
        old_user_summary = self.compute_day_from_reg(old_user_summary)
        
        # union new + old
        updated_summary = old_user_summary.union(new_user_summary)
        
        return updated_summary
      
    # final update
    def update_user_summary(self, week_sum):
        """
            마지막 업데이트
        """
        week_sum = week_sum.createOrReplaceTempView("week_sum")
        week_sum = self.spark.sql("SELECT * FROM week_sum")
        
        # instantiate the update
        user_info = self.df.select(['userId','level','gender','state','churn_service','churn_paid']).distinct()
        updated_summary = week_sum.join(user_info, on=['userId','level'], how='inner')
        
        # load the existing summary 
        user_summary = self.load_summary()
        self.user_summary = user_summary
        
        if self.user_summary.count() >0:
            # compute updated counts 
            new_user_summary = self.update_old_summary(updated_summary)
            self.user_summary = new_user_summary 
            
            # append save 
            self.save_summary('append')
        else:
            # no count, first week?
            
            updated_summary = self.init_user_summary(updated_summary)
            self.user_summary = updated_summary 
            
            self.save_summary('overwrite')