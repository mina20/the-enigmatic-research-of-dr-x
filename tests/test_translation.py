# tests/test_qa.py

import requests

url = "http://127.0.0.1:8000/translate/translate"  

payload = {
    "text": "What is Stem cell?",
    "source_lang": "en",
    "target_lang": "fr",
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())