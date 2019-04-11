import sys
import math
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkConf, SparkContext
from pyspark.sql import *


conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
data = sc.textFile("train.dat")
data2=sc.textFile("test.dat")

def rating_drop(p):
	return (int(p[0]),int(p[1]),float(p[2]))

traindata = data.map(lambda l:l.split('\t'))
train_data=traindata.map(rating_drop).cache()
train_data = train_data.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2])))
testdata = data2.map(lambda x:x.split('\t'))
test_data = testdata.map(lambda p: Row(userId=int(p[0]),movieId=int(p[1])))
'''
def als_tuning(train_x, data, maxIteration,reg):
  min_error = float('inf')
  best_reg = 0
  for j in maxIteration:
    for i in reg:
      als = ALS(maxIter=j, regParam=i, userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="nan")

      model = als.fit(train_x)
      predictions = model.transform(data)
      evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
      rmse = evaluator.evaluate(predictions)
      #print("""""""""RMSE == """, rmse,"Regularisation == ", i) 
      if rmse < min_error:
        min_error = rmse
        best_reg = i
        best_iter = j
      
  return best_reg,best_iter
	
maxIter=[10,15]
regParam=[0.01, 0.155,0.16]
'''
train = spark.createDataFrame(train_data)
train_d, validation_data= train.randomSplit([0.8,0.2])

def model_training(train_data, reg,iterat):
  als = ALS(maxIter=iterat,regParam=reg, userCol="userId",itemCol="movieId",ratingCol="rating",coldStartStrategy="nan")
  model=als.fit(train_data)
  return model

#reg,iterat = als_tuning(train_d,validation_data, maxIter,regParam)
m = model_training(train,reg=0.01,iterat=10)
test=spark.createDataFrame(test_data)
predictions = m.transform(test)
#print("***********predictions",predictions)

pred = predictions.select('prediction')
predict = pred.rdd.map(list)
predict.coalesce(1).saveAsTextFile("submit")
sc.stop()


