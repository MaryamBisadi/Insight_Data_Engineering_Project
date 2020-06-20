from pyspark.sql.types import *
import configparser
from pyspark import SQLContext
from pyspark.sql import SparkSession

# prepare database connection setting from database.ini
def DB_connection():
    DBproperties = {}
    config = configparser.ConfigParser()
    config.read("database.ini")
    dbProp = config['postgresql']
    dbUrl= dbProp['url']
    DBproperties['user']=dbProp['user']
    DBproperties['password']=dbProp['password']
    DBproperties['driver']=dbProp['driver']
    return DBproperties, dbUrl

def Transfer_to_DB(spark, df):
    #Create PySpark DataFrame Schema
    r_schema = StructType([StructField('id',IntegerType(),True)\
                        ,StructField('h1',DoubleType(),True)\
                        ,StructField('h2',DoubleType(),True)\
                        ,StructField('h3',DoubleType(),True)\
                        ,StructField('h4',DoubleType(),True)\
                        ,StructField('h5',DoubleType(),True)\
                        ,StructField('h6',DoubleType(),True)\
                        ,StructField('h7',DoubleType(),True)\
                        ,StructField('h8',DoubleType(),True)\
                        ,StructField('h9',DoubleType(),True)\
                        ,StructField('h10',DoubleType(),True)\
                        ,StructField('h11',DoubleType(),True)\
                        ,StructField('h12',DoubleType(),True)\
                        ,StructField('services',StringType(),True)])

    sqlContext = SQLContext(spark)
    #Create Spark DataFrame from Pandas
    df_record = sqlContext.createDataFrame(df, r_schema)
    #Important to order columns in the same order as the target database
    df_record  = df_record.select("id","h1","h2","h3","h4","h5","h6",\
                                  "h7","h8","h9","h10","h11","h12","services")
    df_record.show()
    properties, url = DB_connection()
    df_record.write.jdbc(url=url,table='patient_records',mode='append',properties=properties)

