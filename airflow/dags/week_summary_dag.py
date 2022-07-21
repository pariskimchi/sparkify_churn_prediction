from datetime import datetime, timedelta
import os

import logging
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.S3_hook import S3Hook
# from airflow.contrib.operators import SparkSubmitOperator

from datetime import datetime, timedelta

# from airflow.operators import DataQualityOperator


default_args = {
    'owner': 'haneul',
    'start_date': datetime(2022, 7, 17),
    'depends_on_past': False, 
    'retries': 3, 
    'retry_delay': timedelta(minutes=5),
    'catchup': False, 
    'email_on_retry': False,
    'aws_conn_id':'aws_conn',
    'redshift_conn_id':'redshift',
    'postgres_conn_id':'redshift',
}
def list_keys():
    hook = S3Hook(aws_conn_id="aws_conn")
    bucket = Variable.get('s3_bucket')
    logging.info(f"Listing Keys from {bucket}")
    keys = hook.list_keys(bucket)
    for key in keys:
        logging.info(f"-s3://{bucket}/{key}")


# 
etl_summary_dag = DAG('week_summary_dag',
          default_args=default_args,
          description='Load and transform data in Redshift with Airflow',
          end_date=datetime(2022,7,30),
          schedule_interval='0 * * * *',
          catchup=False,
          is_paused_upon_creation=False
)

# start operator 
task_start = DummyOperator(
    task_id="start",
    dag=etl_summary_dag
)

# s3 bucket sensor
list_task = PythonOperator(
    task_id="list_keys",
    python_callable=list_keys,
    dag=etl_summary_dag
)

# # pyspark job 실행 
run_etl_cmd = """
    python ~/airflow/dags/etl2.py
"""
# run etl.py BashOperator 
task_etl = BashOperator(
    task_id="etl_week_summary",
    dag=etl_summary_dag,
    bash_command=run_etl_cmd
)

# etl_processing = SparkSubmitOperator(
#         task_id="etl_processing",
#         application="/opt/airflow/dags/etl2.py",
#         conn_id="spark_conn",
#         verbose=False

# )




# s3에 pyspark job 실행결과를 복사한다 
copy_to_s3_cmd ="echo 'copying etl result files to S3' && "

# S3에 airflow log를 복사한다 
copy_log_to_s3_cmd = ""

# task copy to redshift 
copy_to_redshift = BashOperator(
    task_id="copy_to_redshift",
    dag=etl_summary_dag, 
    bash_command="""
        python ~/airflow/dags/copy_redshift.py
    """
)

# redshiftt sensor week_summary as warehouse


# data quality check on redshift
# data_quality_redshift = DataQualityOperator(
#     task_id="data_quality_redshift",
#     dag=etl_summary_dag,
#     redshift_conn_id="redshift",
#     test_query = "SELECT count(userId) from staging_week_summary",
#     expected_result = 53793
# )



# end operator 
task_end = DummyOperator(
    task_id="end",
    dag=etl_summary_dag
)

# task dependencies
task_start >>list_task>> task_etl >> copy_to_redshift >> task_end