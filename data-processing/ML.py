from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest

from pyspark.ml.feature import VectorIndexer
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

def Training(spark):
    #df = spark.read.csv('s3a://mimic3waveforms3/TrainingFeatures.csv',inferSchema=True)
    #df.describe().show()

    data = [LabeledPoint(0.0, [0.0]),LabeledPoint(1.0, [1.0]),LabeledPoint(3.0, [2.0]),LabeledPoint(2.0, [3.0]) ]

    #trainingData = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

    # Train a RandomForest model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    #  Note: Use larger numTrees in practice.
    #  Setting featureSubsetStrategy="auto" lets the algorithm choose.
    model = RandomForest.trainRegressor(sc.parallelize(data), categoricalFeaturesInfo={},\
                                        numTrees=6, featureSubsetStrategy="auto",\
                                        impurity='variance', maxDepth=4, maxBins=32)
    return model

def Prediction(spark, model):
    #df = spark.read.csv('s3a://mimic3waveforms3/TestFeatures/*.csv',inferSchema=True)
    #df = spark.read.csv('s3a://mimic3waveforms3/TestFeatures/TestFeatures_000033.csv',inferSchema=True)
    #df.describe().show()

    data = [LabeledPoint(0.0, [0.0]),LabeledPoint(1.0, [1.0]),LabeledPoint(3.0, [2.0]),LabeledPoint(2.0, [3.0]) ]


    #testData = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))
    predictions = model.predict(sc.parallelize(data).map(lambda x: x.features))
    #print(predictions.collect())
    labelsPredictions = sc.parallelize(data).map(lambda x: x.label).zip(predictions)#testData.map(lambda x: x.label).zip(predictions)
    acc = labelsPredictions.filter(lambda x: x[0] == x[1]).count() / float(testData.count())
    #print(labelsPredictions.collect())

    #labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    #testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum()/float(testData.count())
    print('########### Model Accuracy = ',str(acc)) #+ str(testMSE))
    print('\n###########Learned regression forest model:')
    print(model.toDebugString())
    def main(spark):

    model = Training(spark)
    Prediction(spark, model)

if __name__ == "__main__":
    conf = SparkConf().setMaster("spark://ec2-52-8-238-139.us-west-1.compute.amazonaws.com:7077").setAppName("Mortality Prediction")
    sc = SparkContext(conf = conf)

    spark = SparkSession.builder \
        .master("spark://ec2-52-8-238-139.us-west-1.compute.amazonaws.com:7077") \
        .appName("Mortality Prediction") \
        .getOrCreate()

    SparkSession.builder.config(conf=conf)
    '''
    sqlContext = SQLContext(sc)
    #df = sqlContext.read.format("s3:mimic3waveforms3")
    df = spark.read.csv('s3://"mimic3waveforms3"/PATIENTS.csv',\
        header=True,\
        schema=schema,\
        multiLine=True,\
        quote='"',\
        escape='"')
    
    df.describe().show()
    '''
    # execute only if run as a script
    print("\n************Feature Extraction...\n")
    main(spark)
    print("\n*************Finish.\n")

    spark.stop()
