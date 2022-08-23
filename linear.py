from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
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

df = spark.read.format("csv").schema(schema).option("header", False).load("test_US.csv")

# Splitting the dataframe as Train and Test sets
train, test = df.randomSplit(weights=[0.8,0.2], seed=200)

# Specifying the features that are to be selected
selected = ['total_cases','new_cases','total_deaths','stringency_index','retail_and_recreation','grocery_and_pharmacy','residential','transit_stations','workplaces']

# Transforming train and test set with Vector assembler
assembler = VectorAssembler(inputCols = selected, outputCol = "features")
assemble = assembler.transform(train)
assemble_test = assembler.transform(test)


# Modelling Linear Regressor and fitting the model
lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(assemble)
predictions = lr_model.transform(assemble_test)
predictions.show(3)

# Evaluating the prediction
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
mean_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
r2=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 =r2.evaluate(predictions)
rmse = evaluator.evaluate(predictions)
mse = mean_evaluator.evaluate(predictions)
print("R squared",r2)
print("Root Mean Square Error",rmse/10000)
print("Mean Square Error",mse/10000)

# Plotting the
x_ax = range(0, predictions.count())
y_pred=predictions.select("prediction").collect()
y_orig=predictions.select("label").collect()

plt.plot(x_ax, y_orig, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Real GDP and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 
 