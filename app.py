import logging
import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field,constr, confloat
import pandas as pd
from typing import List



# Create a FastAPI instance
app = FastAPI()



# Define a Pydantic model for the input data
class GradingItem(BaseModel):
    loan_amnt: confloat(ge=0)
    home_ownership: str
    annual_inc: confloat(ge=0)
    purpose: str
    dti: confloat(ge=0)
    total_acc: confloat(ge=0)
    tot_cur_bal: confloat(ge=0)
    acc_open_past_24mths: confloat(ge=0)
    mort_acc: confloat(ge=0)
    employment_length: int

class SubgGradingItem(BaseModel):
    loan_amnt: confloat(ge=0)
    home_ownership: constr()
    annual_inc: confloat(ge=0)
    purpose: constr()
    dti: confloat(ge=0)
    total_acc: confloat(ge=0)
    tot_cur_bal: confloat(ge=0)
    acc_open_past_24mths: confloat(ge=0)
    mort_acc: confloat(ge=0)
    employment_length: int
class AcceptedItem(BaseModel):
    loan_amount: confloat(ge=0)  # Constrained float for loan amount (greater than or equal to 0)
    loan_title: constr()  # Constrained string for loan title
    dti: confloat(ge=0)  # Constrained float for debt-to-income ratio (greater than or equal to 0)
    state: constr()  # Constrained string for state
    employment_length: confloat(ge=0)  # Constrained integer for employment length (greater than or equal to 0)
 




with open('catcls_grade_model.pkl', 'rb') as f:
    cat_grade_model = pickle.load(f)


with open('catcls_rejected_accepted_model.pkl', 'rb') as f:
    cat_accepted_model = pickle.load(f)

with open('catcls_subgrade1.pkl', 'rb') as f:
    sub_grade_model = pickle.load(f)




@app.get("/")
def read_root():
    return {"message": "Welcome to Lending Club"}

@app.post('/loan_accept_reject')
async def scoring_endpoint(item: AcceptedItem):
    try:
        # Convert the Pydantic model to a Pandas DataFrame
        item_dict = item.dict()
        # Convert the Pydantic model to a Pandas DataFrame
        df = pd.DataFrame([item_dict])



        # Make probability predictions using the LightGBM model
        pred_proba = cat_accepted_model.predict_proba(df)

        # Assuming a binary classification problem, use probabilities for the positive class
        positive_class_probability = pred_proba[:, 1]

        
        response = {
            "Probability of getting loan from landingclub is: ": positive_class_probability[0],
        }

        return response

    except Exception as e:
        # Handle exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=str(e))

# Define the scoring endpoint
@app.post('/grading', response_model=dict)
async def grading_endpoint(item: GradingItem):
    try:
        item_dict = item.dict()
        # Convert the Pydantic model to a Pandas DataFrame
        df = pd.DataFrame([item_dict])

        # Make predictions using the LightGBM model
        predicted_class = cat_grade_model.predict(df)

        response = {
            "Predicted grade of your loan is: ": predicted_class[0].tolist(),
        }

        return response

    except Exception as e:
        # Log the exception for debugging
        logging.exception("An error occurred in grading_endpoint:")

        # Handle exceptions and return an HTTP 500 error
        return JSONResponse(status_code=500, content={"detail": str(e)})
    

@app.post('/sub_grading', response_model=dict)
async def grading_endpoint(item: SubgGradingItem):
    try:
        item_dict = item.dict()
        # Convert the Pydantic model to a Pandas DataFrame
        df = pd.DataFrame([item_dict])

        # Make predictions using the LightGBM model
        predicted_class = sub_grade_model.predict(df)

        response = {
            "Predicted sub_grade of your loan is: ": predicted_class[0].tolist(),
        }

        return response

    except Exception as e:
        # Log the exception for debugging
        logging.exception("An error occurred in grading_endpoint:")

        # Handle exceptions and return an HTTP 500 error
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))