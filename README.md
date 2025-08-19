**House Price Prediction API**

This project is part of the **ML Model Inference with FastAPI**. 
It provides a REST API for predicting house prices using a trained machine learning model.

**Problem Description**
We solve the **House Price Prediction** problem:
 **Input**: Features of a house (bedrooms, bathrooms, stories).
 **Output**: Predicted house price along with a confidence interval.

Dataset reference: Housing Prices Dataset
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

 Model Choice
We trained a regression model (
The model was saved as `model.pkl`, and preprocessing (scaling) was saved as `scaler.pkl` using **joblib**.

**API Endpoints**

**1. Health Check**
GET /

**Response**
{
  "status": "Healthy",
  "message": "House Price Prediction API is Running"
}

**3. Prediction**
POST /predict
{
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 1
}

**Response**
{
  "prediction": 125000.0,
  "confidence_interval": [112500.0, 137500.0]
}

**How To Run**

**Clone the repository**
git clone https://github.com/ShayilaMarickar/FastAPI-Application.git
cd FastAPI-Application

**Install Dependencies**
pip install -r requirements.txt

**Start FastAPI Server**
uvicorn main:app --reload

**Open the API in your browser**
http://127.0.0.1:8000/docs


**Project Structure**

├── main.py              
├── model.pkl            
├── scaler.pkl           
├── requirements.txt     
└── README.md            
├── FastAPI.ipynb
