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
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "SupportVectorRegressor": SVR(kernel='rbf', C=100, gamma=0.1),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5)
    }

    best_model = None
    best_rmse = float("inf")
    best_r2 = float("-inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} RMSE: {rmse:.2f}")
        print(f"{name} R² Score: {r2:.2f}\n")
        
        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
            best_r2 = r2
    
    # Save the best model
    if best_model:
        with open("house_price_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
    
    print(f"Best Model Selected: {best_model.__class__.__name__}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"Best R² Score: {best_r2:.2f}")

train_model()
