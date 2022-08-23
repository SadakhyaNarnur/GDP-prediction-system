1. Data Preprocessing and cleaning
On accumulating the whole dataset, it has null values which can be manually removed. 
For missing values one way is adding 0 for intitial values of deaths and new cases as at start of covid there were not many.
For missing stringency index imputer needs to be run on each country to average the stringency_index in that country to fill missing.

imputer_apply.py
------------------
3 csv files needs to be input : owid-covid-data.csv, changes-visitors-covid.csv, gdp-cleaned.csv
Output: In the specified folder a cleaned csv file will be generated.


2. Predicting using Regressors
gradient.py
------------
This takes as input the dataset and provides prediction as well as error measures.
Input: test_US.csv it is the dataset for predicting United States gdp
Output: R squared, Root Mean Square Error, Mean Absolute Error, Plotted graph comparing Predictions and Original expected value.

linear.py
------------
This takes as input the dataset and provides prediction as well as error measures.
Input: test_US.csv it is the dataset for predicting United States gdp
Output: R squared, Root Mean Square Error, Mean Absolute Error, Plotted graph comparing Predictions and Original expected value.

random_forest.py
----------------
This takes as input the dataset and provides prediction as well as error measures.
Input: test_US.csv it is the dataset for predicting United States gdp
Output: R squared, Root Mean Square Error, Mean Absolute Error, Plotted graph comparing Predictions and Original expected value.

main.py
----------------
"Requirement" : sklearn, pandas, numpy
                custom.py needs to be in the same folder as main.py

This takes as input the dataset and provides prediction as well as error measures.
Input: test_US.csv it is the dataset for predicting United States gdp
Output: R squared, Root Mean Square Error, Mean Absolute Error, Plotted graph comparing Predictions and Original expected value.
However for better results it can be hypertuned by changing the maxIteration and layers.

custom.py
-------------------
Internally run or called when RFMLPRegressor is imported from this file.

now_cast_rf.py
------------------
This takes as input the dataset and provides prediction.
Input: now_cast.csv it is the dataset for predicting United States gdp
Output: Predicted value of GDP dataframe will be printed.

data-and-preparation
------------------------
Has 5 files: owid-covid-data.csv, changes-visitors-covid.csv, gdp-cleaned.csv, Cleaned_data.csv, test_US.csv
Cleaned_data.csv - It is the final cleaned and preprocessed dataset