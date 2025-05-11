from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned_csv.csv')

# Create a Pydantic model for the input data
class CarData(BaseModel):
    company: str
    car_model: str
    year: int
    fuel_type: str
    kilo_driven: int

@app.post("/predict")
async def predict(car_data: CarData):
    try:
        # Extract data from the POST request
        company = car_data.company
        car_model = car_data.car_model
        year = car_data.year
        fuel_type = car_data.fuel_type
        kms_driven = car_data.kilo_driven

        # Ensure the input data matches the format expected by the model
        if company not in car['company'].values:
            raise HTTPException(status_code=400, detail="Company not found in data")
        if fuel_type not in car['fuel_type'].values:
            raise HTTPException(status_code=400, detail="Fuel type not found in data")
        
        # Prepare data for prediction
        prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
        
        # Return the result
        result = np.round(prediction[0], 2)
        return {"prediction": result}

    except Exception as e:
        # Log the error
        print(f"Error: {str(e)}")
        # Return an HTTP error response
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# To run the FastAPI app
# uvicorn app:app --reload

