import requests

url = "http://127.0.0.1:8089/predict"
payload = [{"Age": 30,"Annual_Income": 50000, "Credit_Score": 700,
           "Loan_Amount": 15000, "Number_of_Open_Accounts": 5, "Loan_Approval": 0}]
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

result = response.json()
prediction = '✅' if result.get("prediction")[0] == 1 else '❌'
probability = round(100 * result.get("prediction_proba")[0][1], 2)
print(f"Status Code: {response.status_code}")
print(f"Response Body: {prediction} | {probability}")