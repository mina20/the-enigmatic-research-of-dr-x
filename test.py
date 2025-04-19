import requests

url = "http://127.0.0.1:8000/ask"

payload = {
    "question": "What is Stem cell?"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.text)