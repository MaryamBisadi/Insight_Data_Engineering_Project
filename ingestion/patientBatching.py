import featureExtraction
from pyspark import SQLContext
from pyspark.sql import SparkSession
import boto3
import csv
import threading

def batching(spark, i):

    bucket = "mimic3waveforms3"
    p = '1.0/p0'+ str(i) +'/'

    session = boto3.session.Session()
    client = session.client('s3')
    s3 = session.resource('s3')

    patientList = []
    result = client.list_objects(Bucket=bucket, Prefix=p, Delimiter='/')
    for patient in result.get('CommonPrefixes'):
        patientList.append(patient.get('Prefix'))

    my_bucket = s3.Bucket("mimic3waveforms3")
    patientInfoObj = client.get_object(Bucket=bucket, Key='PATIENTS.csv')

    f_schema = StructType([\
    StructField('min', DoubleType(), True),\
    StructField('max', DoubleType(), True),\
    StructField('mean', DoubleType(), True),\
    StructField('median', DoubleType(), True),\
    StructField('mode', DoubleType(), True),\
    StructField('std', DoubleType(), True),\
    StructField('kurtosis', DoubleType(), True),\
    StructField('mortality_flag', IntegerType(), True),\
    StructField('patient_id', IntegerType(), True)])

    sqlContext = SQLContext(spark)

    data = [(0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0, 0)]
    df_features = sqlContext.createDataFrame(data, f_schema)
    df = pd.read_csv(patientInfoObj['Body'])

    # for first 6 batch of records  extract train features
    for patient in patientList:
       patientRecords = []
       for object_summary in my_bucket.objects.filter(Prefix=patient):
          patientRecord = {}
          patientRecord['file_name']=object_summary.key
          if len(patientRecord['file_name']) != 32 or 'layout' in patientRecord['file_name']:
              continue                                                                                                              287,63        72%
          patientRecord['body']=object_summary.get()['Body'].read()
          patientRecords.append(patientRecord)
       patientId =int(patient[-7:-1])
       mortality = (df[df['SUBJECT_ID']==patientId]['EXPIRE_FLAG'])

      # for first 6 batch of records  extract train features
      if i < 6:   
          df_new_features = featureExtraction.TrainFeatureExtraction(sqlContext, patientRecords, f_schema, mortality, patientId)
          df_features = df_features.union(df_new_features)

      # for last 2 batchs of records extract test features
      else: 
          df_new_features = featureExtraction.TestFeatureExtraction(sqlContext, patientRecords, df_features, f_schema, int(mortality.values), patientId)
          df_features = df_features.union(df_new_features)
          df_features = df_features.filter(col('patient_id')>0)
          df_features.write.csv("s3a://"+bucket+"/PredictFeatures/"+str(patientId))

      if i < 6: 
        df_features = df_features.filter(col('patient_id')>0)
        df_features.write.csv("s3a://"+bucket+"/TrainFeatures/"+str(i))

if __name__ == "__main__":

    spark = SparkSession.builder \
                        .master('spark://ec2-52-8-238-139.us-west-1.compute.amazonaws.com:7077') \
                        .appName('Feature Extraction') \
                        .config('spark.executors.memory', '30gb')\
                        .config('spark.executor.cores, 7')\
                        .getOrCreate()
    
    # divided data to 9 batches and let each batch be proccessed by a thread
    for i in range(8):
        t = threading.Thread(target=batching, args=(spark, i))
        t.start()
        print("\n************Feature Extraction thread"+ str(i) +"...\n")
