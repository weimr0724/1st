from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_model():
    # Load labeled dataset
    data = pd.read_csv("House.csv")
    
    # Define features and target variable
    X = data[['1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'BedroomAbvGr', 'GarageArea']]
    y = data["SalePrice"]

    # Select training and test data based on "DataType" column
    X_train = X[data["DataType"] == "Training"]
    y_train = y[data["DataType"] == "Training"]
    X_test = X[data["DataType"] == "Test"]
    y_test = y[data["DataType"] == "Test"]


    # Dictionary of models to try
    '''
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "SupportVectorRegressor": SVR(kernel='rbf', C=100, gamma=0.1),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5)
    }
    '''

    model = LinearRegression()
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    with open("house_price_model.pkl", "wb") as f:
        pickle.dump(model, f)    

train_model()
