import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
print(genai.list_models())
