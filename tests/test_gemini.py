import os 
import google.generativeai as genai

genai.configure(api_key="AIzaSyCjznKUifMfOL3WT26lCIBtKbMemTRIHa8")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("What is Retrieval Augmented Generation?")
print(response.text)
