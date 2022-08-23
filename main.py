from custom import RFMLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():
    # Reading the input dataset and dropping if any missing values
    train = pd.read_csv("test_US.csv")
    train.dropna(inplace=True)

    # Features are selected 
    X = train.iloc[:,8:-1]
    Y = train.iloc[:,-1:]
    
    # Splitting into rain and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=1)

    # Modeling our RFMLRegressor that is taken form custom.py
    # the first parameter takes the number of MLP we want to run and rest arest are MLP specific parameters
    regr = RFMLPRegressor(30,hidden_layer_sizes=(300,600),max_iter = 1000,activation='relu')
    print(f"Regressor created, training with X: {X.shape}, Y: {Y.shape}")
    
    # Fitting the model with Train data
    regr.fit(X_train, y_train)
    print("Predictions:")
    prediction = regr.predict(X_test)
    print(prediction)
    y_test = np.ravel(y_test)
    print(y_test)
    print("R Squared Error: ",r2_score(y_test, prediction, force_finite=False))
    print("Mean Absolute Error:",mean_absolute_error(y_test, prediction)/10000)
    print("Root Mean Square Error:",mean_squared_error(y_test, prediction, squared=False)/10000)
if __name__ == '__main__':
    main()