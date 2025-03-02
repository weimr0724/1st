import joblib, pickle, pandas as pd, numpy as np
from tabpy.tabpy_tools.client import Client

model = joblib.load('house_price_model.pkl')
def predicted_house_price(FirstFlrSF, SecondFlrSF, FullBath, HalfBath, TotRmsAbvGrd, BedroomAbvGr, GarageArea):
    # Load the trained model
    # Prepare input data for prediction
    data = pd.DataFrame({
        '1stFlrSF':FirstFlrSF,
        '2ndFlrSF':SecondFlrSF,
        'FullBath':FullBath,
        'HalfBath':HalfBath,
        'TotRmsAbvGrd':TotRmsAbvGrd,
        'BedroomAbvGr':BedroomAbvGr,
        'GarageArea': GarageArea
    })

    # Make predictions
    return model.predict(data).tolist()

# Connect to TabPy and deploy
client = Client('http://localhost:9004/')
client.deploy("predicted_house_price", predicted_house_price, override=True)
