import TransferToDB
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier
import pandas
  
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
    
    # random forest classification is used for mortality prediction
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
    # for future upgrade, service can be read from other tables of patients database
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
