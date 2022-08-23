from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, lit, collect_list
from graphframes import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, DoubleType

#Creating the spark session named spark
spark = SparkSession.builder.appName("Project-data-cleaning").getOrCreate()

#Impute function to fill the empty values. Applied to only Stringency Index
def apply_imputer_country(country_name):
  inside_data = final_df.filter(final_df.iso_code == country_name)
  imputer = Imputer()
  imputer.setInputCols(["stringency_index"])
  imputer.setOutputCols(["stringency_index_new"])
  model = imputer.fit(inside_data)
  model.setInputCols(["stringency_index"])
  x = model.transform(inside_data)
  return x

#Schema defined for the creating empty datafrome and adding imputed data
empty_df_schema = StructType([
  StructField('year_quarter', StringType(), True),
  StructField('iso_code', StringType(), True),
  StructField('continent', StringType(), True),
  StructField('location', StringType(), True),
  StructField('date', DateType(), True),
  StructField('total_cases', IntegerType(), True),
  StructField('new_cases', IntegerType(), True),
  StructField('total_deaths', IntegerType(), True),
  StructField('stringency_index', DoubleType(), True),
  StructField('retail_and_recreation', DoubleType(), True),
  StructField('grocery_and_pharmacy', DoubleType(), True),
  StructField('residential', DoubleType(), True),
  StructField('transit_stations', DoubleType(), True),
  StructField('parks', DoubleType(), True),
  StructField('workplaces', DoubleType(), True),
  StructField('GDP', DoubleType(), True)
  ])

#Reading strngency Index data
stringency_index_df = spark.read.option("inferSchema","true").option("header","true").csv("C:/Users/Nikitha/OneDrive/Documents/School/Assignments/Big_Data/to-be-submitted/owid-covid-data.csv")
#Reading Google mobility data
google_data_df = spark.read.option("inferSchema","true").option("header","true").csv("C:/Users/Nikitha/OneDrive/Documents/School/Assignments/Big_Data/to-be-submitted/changes-visitors-covid.csv")

#Converting the dataframes to views
stringency_index_df.createOrReplaceTempView("stringency_index_table")
google_data_df.createOrReplaceTempView("google_data_table")

#creating the final feature dataframes
features_df = spark.sql("select iso_code,continent,location,date,total_cases,new_cases,total_deaths,stringency_index,retail_and_recreation,grocery_and_pharmacy, residential, transit_stations, workplaces from stringency_index_table a JOIN google_data_table b ON a.date=b.Day and a.iso_code=b.Code")
#filling the empty values for only the given columns to 0
features_df.na.fill(value=0,subset=["total_cases","new_cases","total_deaths"]).createOrReplaceTempView("features_table")

#Reading the cleaned GDP data
gdp_data = spark.read.option("inferSchema","true").option("header","true").csv("C:/Users/Nikitha/OneDrive/Documents/School/Assignments/Big_Data/to-be-submitted/gdp-cleaned.csv")
gdp_data.createOrReplaceTempView("gdp_data")

#Updating the features data with addion year_quarter so that we can combine GDP data with it
updated_feature_df = spark.sql("select CONCAT(year,quarter) as year_quarter,* from (select *, CASE WHEN a.month IN (1,2,3) then 'Q1' WHEN a.month IN (4,5,6) THEN 'Q2' WHEN a.month IN (7,8,9) THEN 'Q3' ELSE 'Q4' END as quarter from(select split(date,'/')[0] as month, split(date,'/')[2] as year, * from features_table)a)b").drop("month","year","quarter")
updated_feature_df.createOrReplaceTempView("updated_features")

#creation of final GDP and features data together
final_df = spark.sql("select a.*,GDP from updated_features a JOIN gdp_data b ON a.location=b.Country and a.year_quarter=b.quarter")

#Countries where the null values are present for Stringency INdex and wil go into applying imputation
list_to_impute = final_df.filter(final_df.stringency_index.isNull()).select("iso_code").distinct().rdd.flatMap(lambda x: x).collect()

emptyRDD = spark.sparkContext.emptyRDD()
impute_df = spark.createDataFrame(emptyRDD,empty_df_schema)

for i in list_to_impute:
    impute_df = impute_df.union(apply_imputer_country(i).drop("stringency_index").withColumnRenamed("stringency_index_new","stringency_index"))

##---------Remove the comment and provide the country name in the filter below to fetch any specific country to the output
#impute_df = impute_df.filter(impute_df.location == "United States")
#Final Write
impute_df.coalesce(1).write.mode("overwrite").csv("C:/Users/Nikitha/OneDrive/Documents/School/Assignments/Big_Data/to-be-submitted/output-data-clean-model")
