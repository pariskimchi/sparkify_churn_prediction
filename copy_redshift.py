import configparser
import psycopg2
from sql_queries import copy_table_queries,create_table_queries, drop_table_queries


def drop_tables(cur, conn):
    """
        Drops each table using the queries in `drop_table_queries` list.
    """
    for query in drop_table_queries:
        cur.execute(query)
        conn.commit()

def create_tables(cur, conn):
    """
        Creates each table using the queries in `create_table_queries` list. 
    """
    for query in create_table_queries:
        cur.execute(query)
        conn.commit()

def load_staging_tables(cur, conn):
    """
        Copy staging tables from S3 to redshift 
            using queries in `copy_table_queries` list     
    """
    for query in copy_table_queries:
        cur.execute(query)
        conn.commit()



def main():

    """
        - Read AWS-configuration file through configparser
            and set config from `config.cfg` file

        - Creates connection using config

        - Copy staging table from s3 to redshift 
            using `load_staging_tables` function

        - Insert staging table data into Fact-Dimension tables 
            using `insert_tables` function
    """
    config = configparser.ConfigParser()
    config.read('config.cfg')

    conn = psycopg2.connect("host={} dbname={} user={} password={} port={}".format(*config['REDSHIFT'].values()))
    cur = conn.cursor()
    
    drop_tables(cur, conn)
    create_tables(cur, conn)
    load_staging_tables(cur, conn)

    conn.close()


if __name__ == "__main__":
    main()