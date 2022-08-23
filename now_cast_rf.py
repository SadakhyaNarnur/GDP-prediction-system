from cgi import test
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Initializing SparkContext and Sql Session
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# Defining schema for dataframe
schema = StructType([
StructField("Quarter", StringType(), True),
StructField("Code", StringType(), True),
StructField("continent", StringType(), True),
StructField("location", StringType(), True),
StructField("date", StringType(), True),
StructField("total_cases", IntegerType(), True),
StructField("new_cases", IntegerType(), True),
StructField("total_deaths", IntegerType(), True),
StructField("stringency_index", FloatType(), True),
StructField("retail_and_recreation", FloatType(), True),
StructField("grocery_and_pharmacy", FloatType(), True),
StructField("residential", FloatType(), True),
StructField("transit_stations", FloatType(), True),
StructField("parks", FloatType(), True),
StructField("workplaces", FloatType(), True),
StructField("label", FloatType(), True)
])

schema1 = StructType([
StructField("Quarter", StringType(), True),
StructField("Code", StringType(), True),
StructField("continent", StringType(), True),
StructField("location", StringType(), True),
StructField("date", StringType(), True),
StructField("total_cases", IntegerType(), True),
StructField("new_cases", IntegerType(), True),
StructField("total_deaths", IntegerType(), True),
StructField("stringency_index", FloatType(), True),
StructField("retail_and_recreation", FloatType(), True),
StructField("grocery_and_pharmacy", FloatType(), True),
StructField("residential", FloatType(), True),
StructField("transit_stations", FloatType(), True),
StructField("parks", FloatType(), True),
StructField("workplaces", FloatType(), True)
])

train = spark.read.format("csv").schema(schema).option("header", False).load("test_US.csv")
test_df = spark.read.format("csv").schema(schema1).option("header", False).load("now_cast.csv")

# Specifying the features that are to be selected
selected = ['total_cases','new_cases','total_deaths','stringency_index','retail_and_recreation','grocery_and_pharmacy','residential','transit_stations','workplaces']

# Transforming train and test set with Vector assembler
assembler = VectorAssembler(inputCols = selected, outputCol = "features")

# Modelling RandomForestRegressor and fitting the model
rf = RandomForestRegressor(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[assembler, rf])
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
    .build()

# Fitting and transforming with CrossValidation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=50)
cvModel = crossval.fit(train)
predictions = cvModel.transform(test_df)
predictions.show(3)

