import requests
import json
import os
from dotenv import load_dotenv

# Specify the URL and the authentication token
load_dotenv()
url = "https://api.replicate.com/v1/collections/language-models"
auth_token = os.getenv("REPLICATE_AUTH_TOKEN")
headers = {"Authorization": f"Token {auth_token}"}

# Make a GET request to the URL with the headers
response = requests.get(url, headers=headers)

# Ensure the request was successful
if response.status_code == 200:
    # Convert the response to JSON
    data = response.json()

    # Open a file to write the JSON data
    with open("data.json", "w") as file:
        # Write the JSON data to the file
        json.dump(data, file)
else:
    print("Failed to retrieve data")
