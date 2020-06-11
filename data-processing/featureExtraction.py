import os
import boto3
import wfdb, wfdb.processing
import math
import ntpath
import numpy as np
from scipy import stats
import pandas as pd
import csv

def readFromS3(bucket, prefix):
    patientList = []
    s3 = boto3.client('s3')
    result = s3.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    for patient in result.get('CommonPrefixes'):
        patientList.append(patient.get('Prefix'))

    return patientList

def writeTmpFile(record):
    try:
        f = open(record['file_name'][-16:], "wb")
        f.write(record['body'])
        f.close()
    except:
        print("can not write the temp record!")

def deleteTmpFile(record):
    #path = os.getcwd()
    try:
        os.remove(record)#path+record)#file name is main.py=7
        #os.remove(path+record)
    except:
        print(record)
        print("can not remove the temp record")

for i in range(0, len(patientRecords), 2):

        if len(patientRecords[i]['file_name']) == 32 and 'layout' not in patientRecords[i]['file_name']:
            #print(">>>>>>> ",patientRecords[i]['file_name'][-16:])
            #print(">>>>>>> ",patientRecords[i+1]['file_name'][-16:]) 

            writeTmpFile(patientRecords[i])
            if  'layout' not in patientRecords[i+1]['file_name']:
                writeTmpFile(patientRecords[i+1])

            signals, fields = wfdb.rdsamp(patientRecords[i]['file_name'][-16:-4])

            deleteTmpFile(patientRecords[i]['file_name'][-16:])
            if  'layout' not in patientRecords[i+1]['file_name']:
                deleteTmpFile(patientRecords[i+1]['file_name'][-16:])

            if fields['sig_len'] < 200 or fields['sig_len'] > 100000:# > 10000 for test
                continue

            # preprocessing: nan to 0 for missing signals in the wave
            for channel in range(len(signals[0])):
                signals[:,channel] = [0 if math.isnan(x) else x for x in signals[:,channel]] # preprocessing: nan -> 0

            # get HR
            qrs_inds = wfdb.processing.gqrs_detect(sig=signals[:,0], fs=fields['fs'])
            heart_rate_wfdb=wfdb.processing.compute_hr(fields['sig_len'],qrs_inds,fields['fs'])

            # preprocessing: nan to 0 for missing signals in HR
            heart_rate_wfdb = [0 if math.isnan(x) else x for x in heart_rate_wfdb] # preprocessing: nan -> 0


            #print(seg_count, ' ... ',len(heart_rate_wfdb))
            heart_rate+=heart_rate_wfdb

            if len(heart_rate_wfdb)>200:
                seg_count+=1
            #for test
            if seg_count > 3:
                break

    if len(heart_rate)>200:
        f_min = round(np.amin(heart_rate)*100)/100
        f_max = round(np.amax(heart_rate)*100)/100
        f_mean = round(np.mean(heart_rate)*100)/100
        f_median = round(np.median(heart_rate)*100)/100
        f_mode = round(stats.mode(heart_rate)[0][0]*100)/100
        f_std = round(np.std(heart_rate)*100)/100
        f_kurtosis = round(stats.kurtosis(heart_rate, fisher=True)*100)/100

    return  [f_min, f_max, f_mean, f_median, f_mode, f_std, f_kurtosis]

def TestFeatureExtraction(patientRecords,segmentLength):
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
    for i in range(0, len(patientRecords), 2):

            #if len(patientRecords[i]['file_name']) == 32 and 'layout' not in patientRecords[i]['file_name']:    
            writeTmpFile(patientRecords[i])
            if  'layout' not in patientRecords[i+1]['file_name']:
                writeTmpFile(patientRecords[i+1])

            signals, fields = wfdb.rdsamp(patientRecords[i]['file_name'][-16:-4])#'p000030/3524877_0001')#, sampfrom=8000, sampto=10000)#, channel_names =['PLETH','RESP'])

            deleteTmpFile(patientRecords[i]['file_name'][-16:])
            if  'layout' not in patientRecords[i+1]['file_name']:
                deleteTmpFile(patientRecords[i+1]['file_name'][-16:])

            if fields['sig_len'] < 200 or fields['sig_len'] > 100000:# > 10000 for test
                continue

            # preprocessing: nan to 0 for missing signals in the wave
            for channel in range(len(signals[0])):
                signals[:,channel] = [0 if math.isnan(x) else x for x in signals[:,channel]] # preprocessing: nan -> 0

            # get HR
            qrs_inds = wfdb.processing.gqrs_detect(sig=signals[:,0], fs=fields['fs'])
            heart_rate_wfdb=wfdb.processing.compute_hr(fields['sig_len'],qrs_inds,fields['fs'])

            # preprocessing: nan to 0 for missing signals in HR
            heart_rate_wfdb = [0 if math.isnan(x) else x for x in heart_rate_wfdb] # preprocessing: nan -> 0


            #print(seg_count, ' ... ',len(heart_rate_wfdb))
            heart_rate+=heart_rate_wfdb

            if len(heart_rate_wfdb)>200:
                seg_count+=1

            #for test
            if seg_count > 3:
                break

    if len(heart_rate)>200 and len(heart_rate)>7500*segmentLength:
        HR_seg = heart_rate[:7500*segmentLength]
        f_min = round(np.amin(HR_seg)*100)/100
        f_max = round(np.amax(HR_seg)*100)/100
        f_mean = round(np.mean(HR_seg)*100)/100
        f_median = round(np.median(HR_seg)*100)/100
        f_mode = round(stats.mode(HR_seg)[0][0]*100)/100
        f_std = round(np.std(HR_seg)*100)/100
        f_kurtosis = round(stats.kurtosis(HR_seg, fisher=True)*100)/100


    return  [f_min, f_max, f_mean, f_median, f_mode, f_std, f_kurtosis]
    
    def saveToCSV(featureList, fileName):
    file = open(fileName, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(featureList)

def readFromCSV(fileName):
    featureList = []
    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            featureList.append(row)
    return featureList

def main():

    bucket = "mimic3waveforms3"
    p = '1.0/p00/'

    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket("mimic3waveforms3")
    patientRecords = []
    patientList = readFromS3(bucket, p)

    # Ectract feature for training
    features_train = []

    client = boto3.client('s3')
    patientInfoObj = client.get_object(Bucket=bucket, Key='PATIENTS.csv')
    #body = patientInfoObj['Body']
    #patientInfo = body.read().decode('utf-8')
    #df = pd.read_csv(StringIO(patientInfo))

    pn=0

    df = pd.read_csv(patientInfoObj['Body'])
    for patient in patientList:
        for object_summary in my_bucket.objects.filter(Prefix=patient):
            patientRecord = {}
            #patientRecord['patient_ID'] = patient[-7:-1]
            patientRecord['file_name']=object_summary.key
            #print(patientRecord['file_name'])
            patientRecord['body']=object_summary.get()['Body'].read()
            patientRecords.append(patientRecord)
            #print("s3a://mimic3waveforms3"+str(patientList[0])+"*.hea")
            #patientRecords = sc.textFile("s3a://mimic3waveforms3/"+str(patientList[0])+"*.hea")
            #print(patientRecords_key)
            print("Patient Record:",patientRecord['file_name'])
        patientId =int(patient[-7:-1])
        mortality = (df[df['SUBJECT_ID']==patientId]['EXPIRE_FLAG'])
        extractedFeatures = TrainFeatureExtraction(patientRecords)
        extractedFeatures.append(int(mortality.values))
        extractedFeatures.append(patientId)
        if extractedFeatures[0]!=-1:
            features_train.append(extractedFeatures)

        pn+=1
        if pn>2:
            pn=0
            break

    saveToCSV(features_train, 'TrainFeatures/TrainingFeatures.csv')

    print("**********Extract feature for prediction")
    # Extract feature for prediction
    for patient in patientList:
        features_prediction = []
        for object_summary in my_bucket.objects.filter(Prefix=patient):
            patientRecord = {}
            patientRecord['file_name']=object_summary.key
            if len(patientRecord['file_name']) != 32 or 'layout' in patientRecord['file_name']:
                continue
            patientRecord['body']=object_summary.get()['Body'].read()
            patientRecords.append(patientRecord)
            patientId = int(patient[-7:-1])
            mortality = (df[df['SUBJECT_ID']==patientId]['EXPIRE_FLAG'])
            for segmentCount in range(1,10):
                extractedFeatures = TestFeatureExtraction(patientRecords,segmentCount)
                extractedFeatures.append(int(mortality.values))
                print("...",extractedFeatures)
                if extractedFeatures[0]!=-1:
                    features_prediction.append(extractedFeatures)
                if extractedFeatures[0]==-1:
                    break
                print("Patient Record:",patientRecord['file_name'])
                if extractedFeatures[0]!=-1:
                    features_prediction.append(extractedFeatures)
        if len(features_prediction)>0:
            saveToCSV(features_prediction, 'TestFeaturet/TestFeatures_'+patient[-7:-1]+'.csv')
        pn+=1
        if pn>3:
            break


    print("***********************************")

if __name__ == "__main__":

    print("\n************Feature Extraction...\n")
    main()
    print("\n*************Finish.\n")

                                                                                                    1,9           Top
