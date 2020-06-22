from pyspark import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql.functions import kurtosis,col
from pyspark.sql.types import *
import os
import boto3
import wfdb, wfdb.processing
import math
import numpy as np
import pandas as pd

# read patinet records from S3
def readFromS3(bucket, prefix):
    patientList = []
    s3 = boto3.client('s3')
    result = s3.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    for patient in result.get('CommonPrefixes'):
        patientList.append(patient.get('Prefix'))
    return patientList

# write signal file and its header file to current path
def writeTmpFile(record):
    try:
        f = open(record['file_name'][-16:], "wb")
        f.write(record['body'])
        f.close()
    except:
        print("can not write the temp record!")
# delete the files after HR extraction
def deleteTmpFile(record):
    try:
        os.remove(record)
    except:
        print(record)
        print("can not remove the temp record")
        
# extract training features from heart signals
def TrainFeatureExtraction(sqlContext, patientRecords, f_schema, mortality, patientId):
    f_min=-1
    f_max=-1
    f_mean=-1
    f_median=-1
    f_mode=-1
    f_std=-1
    f_kurtosis=-1
    seg_count=0
    heart_rate = []
    i=0

    data = [(0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0, 0)]
    df_features = sqlContext.createDataFrame(data, f_schema)

    for i in range(0, len(patientRecords), 2):
        if len(patientRecords[i]['file_name']) == 32 and 'layout' not in patientRecords[i]['file_name']:
            writeTmpFile(patientRecords[i])
            if  'layout' not in patientRecords[i+1]['file_name']:
                writeTmpFile(patientRecords[i+1])

            signals, fields = wfdb.rdsamp(patientRecords[i]['file_name'][-16:-4])

            deleteTmpFile(patientRecords[i]['file_name'][-16:])
            if  'layout' not in patientRecords[i+1]['file_name']:
                deleteTmpFile(patientRecords[i+1]['file_name'][-16:])

            if fields['sig_len'] < 200:
                continue

            # preprocessing: nan to 0 for missing signals in the wave
            for channel in range(len(signals[0])):
                signals[:,channel] = [0.0 if math.isnan(x) else x for x in signals[:,channel]] # preprocessing: nan -> 0

            # get HR
            qrs_inds = wfdb.processing.gqrs_detect(sig=signals[:,0], fs=fields['fs'])
            heart_rate_wfdb=wfdb.processing.compute_hr(fields['sig_len'],qrs_inds,fields['fs'])

            # preprocessing: nan to 0 for missing signals in HR
            heart_rate_wfdb = [0.0 if math.isnan(x) else x for x in heart_rate_wfdb] # preprocessing: nan -> 0

            heart_rate+=heart_rate_wfdb
                
    # training feature extraction
    if len(heart_rate)>200:
        hr_schema = StructType([StructField("hr", FloatType(), True)])
        data = {'hr':heart_rate}
        df = pd.DataFrame(data, columns = ['hr'])
        df_heart_rate= sqlContext.createDataFrame(df, hr_schema)
        f_min = round(df_heart_rate.groupBy().min('hr').collect()[0]['min(hr)']*100)/100
        f_max = round(df_heart_rate.groupBy().max('hr').collect()[0]['max(hr)']*100)/100
        f_mean = round(df_heart_rate.groupBy().agg({'hr':'mean'}).collect()[0]['avg(hr)']*100)/100
        f_std = round(df_heart_rate.groupBy().agg({'hr':'stddev'}).collect()[0]['stddev(hr)']*100)/100
        f_median = round(df_heart_rate.approxQuantile('hr', [0.5], 0.002)[0]*100)/100
        df_tmp =df_heart_rate.groupBy('hr').count()
        f_mode = round(df_tmp.orderBy(df_tmp['count'].desc()).collect()[0][0]*100)/100
        f_kurtosis = df_heart_rate.select(kurtosis(df_heart_rate['hr'])).collect()[0]['kurtosis(hr)']
        if math.isnan(f_kurtosis) == True:
            f_kurtosis = 0.0
        else:
            round(f_kurtosis*100)/100
        f_data = {'min':f_min, 'max':f_max, 'mean':f_mean, 'median':f_median, 'mode':f_mode,\
                'std':f_std, 'kurtosis':f_kurtosis, 'mortality':mortality,'patient_id': patientId}
        df_tmp = pd.DataFrame(f_data, columns = ['min','max','mean','median','mode',\
                            'std','kurtosis', 'mortality','patient_id'])
        df_features = sqlContext.createDataFrame(df_tmp, f_schema)
    return  df_features


def TestFeatureExtraction(sqlContext, patientRecords, df_features, f_schema, mortality, patientId): #,segmentLength):
    f_min=-1
    f_max=-1
    f_mean=-1
    f_median=-1
    f_mode=-1
    f_std=-1
    f_kurtosis=-1
    seg_count=0
    heart_rate = []
    i=0

    data = [(0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0, 0)]
    df_features = sqlContext.createDataFrame(data, f_schema)
    for i in range(0, len(patientRecords), 2):
            if '.hea' or '.dat' in patientRecords[i]['file_name']:
                writeTmpFile(patientRecords[i])
            if '.hea' or '.dat' in patientRecords[i+1]['file_name']:
                writeTmpFile(patientRecords[i+1])

            signals, fields = wfdb.rdsamp(patientRecords[i]['file_name'][-16:-4])
            deleteTmpFile(patientRecords[i]['file_name'][-16:])
            if  'layout' not in patientRecords[i+1]['file_name']:
                deleteTmpFile(patientRecords[i+1]['file_name'][-16:])

            if fields['sig_len'] < 200:
                continue

            # preprocessing: nan to 0 for missing signals in the wave
            for channel in range(len(signals[0])):
                signals[:,channel] = [0.0 if math.isnan(x) else x for x in signals[:,channel]] # preprocessing: nan -> 0

            # get HR
            qrs_inds = wfdb.processing.gqrs_detect(sig=signals[:,0], fs=fields['fs'])
            heart_rate_wfdb=wfdb.processing.compute_hr(fields['sig_len'],qrs_inds,fields['fs'])

            # preprocessing: nan to 0 for missing signals in HR
            heart_rate_wfdb = [0.0 if math.isnan(x) else x for x in heart_rate_wfdb] # preprocessing: nan -> 0

            heart_rate+=heart_rate_wfdb

            period = 450000
            if len(heart_rate) > period * 15:
                break
    # test feature extraction for first 12 hours of patient satied in ICU 
    if len(heart_rate)>200: 
        hr_schema = StructType([StructField("hr", FloatType(), True)])
        for h in range(1,13):
            if len(heart_rate) > period*h:
                HR_seg = heart_rate[:period*h]
                data = {'hr':HR_seg}
                df = pd.DataFrame(data, columns = ['hr'])
                df_heart_rate= sqlContext.createDataFrame(df, hr_schema)
                f_min = round(df_heart_rate.groupBy().min('hr').collect()[0]['min(hr)']*100)/100
                f_max = round(df_heart_rate.groupBy().max('hr').collect()[0]['max(hr)']*100)/100
                f_mean = round(df_heart_rate.groupBy().agg({'hr':'mean'}).collect()[0]['avg(hr)']*100)/100
                f_std = round(df_heart_rate.groupBy().agg({'hr':'stddev'}).collect()[0]['stddev(hr)']*100)/100
                f_median = round(df_heart_rate.approxQuantile('hr', [0.5], 0.002)[0]*100)/100
                df_tmp =df_heart_rate.groupBy('hr').count()
                f_mode = round(df_tmp.orderBy(df_tmp['count'].desc()).collect()[0][0]*100)/100
                f_kurtosis = df_heart_rate.select(kurtosis(df_heart_rate['hr'])).collect()[0]['kurtosis(hr)']
                if math.isnan(f_kurtosis) == True:
                    f_kurtosis = 0.0
                else:
                    round(f_kurtosis*100)/100
            else:
                f_min, f_max, f_mean, f_median, f_mode, f_std, f_kurtosis = -1, -1, -1, -1, -1, -1, -1

            f_data = {'min':f_min, 'max':f_max, 'mean':f_mean, 'median':f_median, 'mode':f_mode,\
                    'std':f_std, 'kurtosis':f_kurtosis, 'mortality':mortality,'patient_id': patientId}
            df_tmp = pd.DataFrame(f_data, columns = ['min','max','mean','median','mode',\
                    'std','kurtosis', 'mortality','patient_id'], index=f_data)
            df_new_features = sqlContext.createDataFrame(df_tmp, f_schema)
            df_features = df_features.union(df_new_features)


    return  df_features
                                                                                                                                                                                                
