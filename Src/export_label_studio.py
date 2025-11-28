import requests
import json

API_URL = "Your API URL"
PROJECT_ID = 3
API_TOKEN = "API TOKEN"  

headers = {
    "Authorization": f"Token {API_TOKEN}"
}

# interpolate_key_frames=true
url = f"{API_URL}/api/projects/{PROJECT_ID}/export?interpolate_key_frames=true"

print("Request URL:", url)

response = requests.get(url, headers=headers)

print("Status code:", response.status_code)

if response.status_code != 200:
    print("Error:", response.text)
    exit()

# JSON
with open("project3_interpolated.json", "w", encoding="utf-8") as f:
    f.write(response.text)

print("\n Export to project3_interpolated.json")
