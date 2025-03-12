import requests
from openai import OpenAI
from ..config import OPENAI_API_KEY, SERPAPI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_mitigation_strategies(bias_type):
    """Fetch and structure mitigation strategies for a given bias type."""
    try:
        # Construct search query
        query = f"Mitigation strategies for {bias_type} bias"

        # Search using SerpAPI
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 5
        }

        # Fetch results
        response = requests.get(url, params=params)
        search_results = response.json()

        # Extract snippets
        raw_strategies = []
        for result in search_results.get("organic_results", []):
            snippet = result.get("snippet")
            if snippet:
                raw_strategies.append(snippet)

        raw_text = " ".join(raw_strategies)
        if not raw_text:
            return "No specific mitigation strategies found. Please conduct a manual search."

        # Structure strategies using GPT-3.5
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent assistant that converts raw text into a structured and point-wise list of mitigation strategies. "
                        "Provide clear, organized points for the given bias type."
                    )
                },
                {
                    "role": "user",
                    "content": f"Convert the following raw text into a structured and point-wise list of mitigation strategies for {bias_type} bias:\n\n{raw_text}\n\nPoints:"
                }
            ],
            max_tokens=300,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error fetching mitigation strategies: {str(e)}"