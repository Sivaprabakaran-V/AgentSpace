import os
import requests
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_url:str, mock:bool = False ):
    ''' Manually scrape the information from the LinkedIn profile'''
    if mock:
        linkedin_url = "https://gist.github.com/Sivaprabakaran-V/e4286542b0b81f1042411dd53e1f7d48.js"
        response = requests.get(linkedin_url, timeout=10)

    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ["SCRAPIN_API_KEY"],
            "linkedInUrl": linkedin_url
        }
        response = requests.get(api_endpoint, params=params, timeout=10)

    data = response.json().get("person")
    data = {
        k:v for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["certifications"]
            }
    return data 

if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_url="https://www.linkedin.com/in/alliekmiller/"
        ),
    )