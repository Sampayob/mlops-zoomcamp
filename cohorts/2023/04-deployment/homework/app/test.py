import requests

inference = {
    "year": '2022',
    "month": '04'
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=inference)
print(response.json())