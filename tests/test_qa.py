# tests/test_qa.py

import requests

url = "http://127.0.0.1:8000/qa/ask"  # updated route with /qa prefix if using router

payload = {
    "question": "What is Stem cell?"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())