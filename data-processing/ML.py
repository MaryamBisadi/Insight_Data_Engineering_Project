from pyspark.sql.types import *
import configparser
from pyspark import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier
import pandas

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

    
def ML_Module(spark):

    bucket_train = 's3a://mimic3waveforms3/TrainFeatures/*.csv'#'s3a://mimic3waveforms3/TrainingFeatures.csv'
    bucket_predict = 's3a://mimic3waveforms3/PredictFeatures/*'#'s3a://mimic3waveforms3/TestFeatures/*.csv'

    # Create dataFrame for training features
    df_train = spark.read.csv(bucket_train, inferSchema = True).toDF('MIN',\
                    'MAX',\
                    'MEAN',\
                    'MEDIAN',\
                    'MODE',\
                    'STD',\
                    'KURTOSIS',\
                    'label',\
                    'PATIENTID').cache()
    vector_assembler = VectorAssembler(inputCols=['MIN','MAX','MEAN','MEDIAN','MODE','STD','KURTOSIS'],outputCol="features")
    df_train_features = vector_assembler.transform(df_train).drop('MIN','MAX','MEAN','MEDIAN','MODE','STD','KURTOSIS','PATIENTID')
    #df_train_features = df_temp.drop('MIN','MAX','MEAN','MEDIAN','MODE','STD','KURTOSIS','PATIENTID')
    print('\n Train Features: ', df_train_features.count)
    df_train_features.show(df_train_features.count(),False)
    print("++++++++++++++++++++++Prediction+++++++++++++++++++++++")
    # Create dataFrame for prediction features
    df_predict = spark.read.csv(bucket_predict, inferSchema = True).toDF('MIN',\
                    'MAX',\
                    'MEAN',\
                    'MEDIAN',\
                    'MODE',\
                    'STD',\
                    'KURTOSIS',\
                    'label',\
                    'PATIENTID').cache()
    df_predict = df_predict.filter('PATIENTID>0')
    df_predict_features = vector_assembler.transform(df_predict).drop('MIN','MAX','MEAN','MEDIAN','MODE','STD','KURTOSIS')
    df_predict_features.show()
    print('\n Prediction Features: ', df_predict_features.count)
    df_predict_features.show(df_predict_features.count(),False)

    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    model = rf.fit(df_train_features)

    predictions = model.transform(df_predict_features)
    predictions.show(predictions.count(),False)

    results = predictions.select('PATIENTID','probability')
    df = results.toPandas()
    df['probability'] = df['probability'].apply(lambda col: col[0])

    df2 = df.groupby(df['PATIENTID'])['probability'].apply(list).reset_index(name='h')
    for i in range(12):
        df2['h'+str(i+1)] = df2['h'].apply(lambda col: col[i])
        df2['h'+str(i+1)] = df2['h'+str(i+1)].apply(lambda x: round(x*100)/100)
    del df2['h']

    df2['services'] = 'Test'
    df2.rename(columns={'PATIENTID':'id'}, inplace = True)
    return df2

if __name__ == "__main__":

    spark = SparkSession.builder \
                        .master('spark://ec2-13-57-198-186.us-west-1.compute.amazonaws.com:7077') \
                        .appName('Mortality Prediction') \
                        .getOrCreate()

    df = ML_Module(spark)
    Transfer_to_DB(spark, df)

spark.stop()


                                                                                                                                                                                                  118,1         68%
