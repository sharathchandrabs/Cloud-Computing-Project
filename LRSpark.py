import sys
import numpy as np 
from pyspark import SparkContext
from numpy import *
from numpy.linalg import inv
from pyspark.sql import SparkSession



spark = SparkSession.builder.appName('KMeans').getOrCreate()

data= spark.read.csv('bank.csv',inferSchema=True,header=True,sep=';')
data = data.drop('day')
data = data.drop('month')


def ConvertCategoricaltoNumeric(Dataframe,column):
    indexer = StringIndexer(inputCol=column, outputCol= "Numeric" + column)
    indexed = indexer.fit(Dataframe).transform(Dataframe)
    encoder = OneHotEncoder(inputCol="Numeric" + column, outputCol="categoryVec" + column)
    encoded = encoder.transform(indexed)
    return encoded
categoricalcolumns = []
for name,coltype in data.dtypes:
    if coltype == 'string':
        categoricalcolumns.append(name)

for name in categoricalcolumns:
...     data = ConvertCategoricaltoNumeric(data,name)

>>> data = data.drop('job')
>>> data = data.drop('marital')
>>> data = data.drop('education')
>>> data = data.drop('defualt')
>>> data = data.drop('housing')
>>> data = data.drop('loan')
>>> data = data.drop('contact')
>>> data = data.drop('poutcome')
>>> data = data.drop('y')
>>> data = data.drop('categoryVecjob')
>>> data = data.drop('categoryVecmarital')
>>> data = data.drop('categoryVeceducation')
>>> data = data.drop('categoryVecdefault')
>>> data = data.drop('categoryVechousing')
>>> data = data.drop('categoryVecloan')
>>> data = data.drop('categoryVeccontact')
>>> data = data.drop('categoryVecpoutcome')
>>> data = data.drop('categoryVecy')



(train, test) = final_data.randomSplit([0.7, 0.3])




# df = data.select('Numericy').collect()
# y_list = []
# for l in df:
# 	y_list.append(l[0])

tr = train.rdd
tst = test.rdd
