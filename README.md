# Housing Price Prediction API

A beginner-level machine learning API that serves a trained Scikit-learn housing price prediction model using **FastAPI**.

The project demonstrates how a trained ML model can be exposed through a REST API so that users or other applications can send input data and receive predictions.

##  Project Overview

The machine learning model was trained using the **California Housing dataset**. After training, the model was saved and then integrated with FastAPI.

The API accepts housing-related features through a `POST` request, performs the required preprocessing, passes the data to the trained model, and returns the predicted house value.

### Workflow


Housing Dataset
       ↓
Data Preprocessing
       ↓
Model Training
       ↓
Trained Scikit-learn Model
       ↓
Save Model
       ↓
FastAPI
       ↓
POST /predict
       ↓
Input Validation with Pydantic
       ↓
Preprocessing
       ↓
Model Prediction
       ↓
JSON Response


##  Technologies Used

* Python
* Scikit-learn
* FastAPI
* Uvicorn
* Pydantic
* Pandas
* Joblib / Pickle
* Jupyter Notebook

## What I Implemented

* Trained a Scikit-learn machine learning model on housing data.
* Saved the trained model using model serialization.
* Built a REST API using FastAPI.
* Created a /predict endpoint for making predictions.
* Used **Pydantic** schemas to validate API input.
* Replicated the required **one-hot encoding** preprocessing during inference.
* Tested the API using FastAPI's automatically generated **Swagger UI**.
* Verified that the API successfully returned a model prediction.

##  Project Structure

endtoendml
housing-price-api/
│
├── main.py
├── house_price_model.joblib
├── requirements.txt
├── README.md
|_ eda.ipynb

##  Installation
Clone the repository:
git clone <https://github.com/NVyshnavi007/endtoendml>
cd housing-price-api

Install the required dependencies:
pip install -r requirements.txt

## Running the API

Start the FastAPI application using Uvicorn:
uvicorn main:app --reload
The API will run locally at:
http://127.0.0.1:8000
## 📖 API Documentation

FastAPI provides interactive API documentation automatically.
Open:
http://127.0.0.1:8000/docs
The Swagger UI can be used to enter housing information and test the `/predict` endpoint directly from the browser.

## Prediction Endpoint
POST /predict
The endpoint accepts housing information as JSON input.
Example:
{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.32,
    "ocean_proximity": "NEAR BAY"
}
The API validates the input, performs the required preprocessing, and passes the processed data to the trained model.

Example response:
{
    "prediction": 448057.5238095238
}



##Categorical Encoding

The housing dataset contains the categorical feature `ocean_proximity`.

During training, this categorical feature was converted into numerical features using **one-hot encoding**.

Because the model expects the same feature structure during prediction, the API reproduces the required encoding before passing the input to the model.

This ensures that the input given to the API has the format expected by the trained model.

## Testing

The API was tested locally using FastAPI's Swagger UI.

The /predict endpoint was successfully tested with housing input data and returned a prediction from the trained machine learning model.

##  Current Limitations

This project is intended as a hands-on demonstration of serving a machine learning model through FastAPI.

Currently:

* The API runs locally and is **not publicly deployed**.
* Authentication has not been implemented.
* Advanced logging has not been implemented.
* Comprehensive automated tests have not been added.
* Production-level error handling has not been implemented.
* Model versioning has not been implemented.
* The preprocessing is implemented directly in the API rather than through a complete Scikit-learn preprocessing pipeline.

##  Learning Outcomes

Through this project, I gained practical experience with:

* Serving a trained ML model through a REST API.
* FastAPI fundamentals.
* GET and POST API endpoints.
* Pydantic request validation.
* Model serialization using Pickle/Joblib.
* Applying preprocessing during inference.
* Testing APIs using Swagger UI.
* Understanding the connection between machine learning models and backend APIs.

##  Possible Future Improvements

* Create a complete Scikit-learn preprocessing pipeline.
* Improve validation and error handling.
* Add automated API tests.
* Containerize the application using Docker.
* Deploy the API publicly.
* Add model versioning.
* Add logging and monitoring.

##  Author

N .Naga Vyshnavi

B.Tech Information Technology Student

This project was created as a hands-on implementation of machine learning model serving using FastAPI.
