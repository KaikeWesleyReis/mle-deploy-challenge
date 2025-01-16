import requests
import streamlit as st
from app.variables import LIM_MIN_AGE, LIM_MAX_AGE, LIM_MIN_INCOME, LIM_MIN_LOAN, LIM_MIN_OPEN_ACCOUNTS

URL = "http://localhost:8089/predict"
HEADERS = {"Content-Type": "application/json"}

def main():
    # Streamlit app title
    st.title("Loan Approval Prediction")

    # Sidebar for user inputs
    st.sidebar.header("Input Features")

    # Input fields
    age = st.sidebar.number_input("Age", min_value=LIM_MIN_AGE, max_value=LIM_MAX_AGE, value=30, step=1)
    annual_income = st.sidebar.number_input("Annual Income (in USD)", min_value=LIM_MIN_INCOME, value=50000, step=500)
    credit_score = st.sidebar.number_input("Credit Score", value=700)
    loan_amount = st.sidebar.number_input("Loan Amount (in USD)", min_value=LIM_MIN_LOAN, value=15000, step=500)
    num_open_accounts = st.sidebar.number_input("Number of Open Accounts", min_value=LIM_MIN_OPEN_ACCOUNTS, max_value=50, value=5, step=1)

    # Predict button
    if st.sidebar.button("Predict"):
        # Get prediction through request
        payload = [{"Age": age,"Annual_Income": annual_income, "Credit_Score": credit_score,
                    "Loan_Amount": loan_amount, "Number_of_Open_Accounts": num_open_accounts, "Loan_Approval": 0}]
        
        response = requests.post(URL, json=payload, headers=HEADERS)
        if response.status_code == 200:
            result = response.json()
            prediction = '✅' if result.get("prediction")[0] == 1 else '❌'
            probability = round(100 * result.get("prediction_proba")[0][1], 2)
            # Display results
            st.write(f"**Prediction**: {prediction}")
            st.write(f"**Probability**: {probability} %")
        else:
            st.write('ERROR 404 - Fail on Inference Server.')        
    else:
        st.write("Use the sidebar to input values and click 'Predict'.")

if __name__ == "__main__":
    main()