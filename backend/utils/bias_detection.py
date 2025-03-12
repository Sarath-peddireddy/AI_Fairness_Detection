import re
from openai import OpenAI
from ..config import OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def infer_bias_type(explanation):
    """Infer the type of bias from an explanation using GPT-4."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4",  # Changed from gpt-4o to gpt-4
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent assistant that specializes in identifying bias types. "
                        "Extract the bias type from the given explanation, and provide only the bias type as output. "
                        "If the explanation does not clearly state a bias type, infer the most relevant type."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract the bias type from the following explanation: \"{explanation}\""
                }
            ]
        )
        bias_type = completion.choices[0].message.content.strip()
        return bias_type
    except Exception as e:
        print(f"Error while inferring bias type: {e}")
        return "General Bias"

def detect_bias(text):
    """Detect bias in text using OpenAI."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent assistant that detects bias in text, provides an accurate bias label (Biased/Unbiased), "
                        "most accurate bias score between 0 and 1, type of bias (if known), and a brief explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analyze the following text for bias, label, score, type, and explanation. Text: \"{text}\""
                }
            ]
        )

        response = completion.choices[0].message.content.strip()
        
        # Extract information using simple string parsing
        bias_label = "Unknown"
        if "Bias Label:" in response:
            bias_label = response.split("Bias Label:")[1].split("\n")[0].strip()
        elif "Label:" in response:
            bias_label = response.split("Label:")[1].split("\n")[0].strip()
            
        bias_score = 0.0
        if "Bias Score:" in response:
            try:
                bias_score = float(response.split("Bias Score:")[1].split("\n")[0].strip())
            except:
                pass
        elif "Score:" in response:
            try:
                bias_score = float(response.split("Score:")[1].split("\n")[0].strip())
            except:
                pass
                
        bias_type = "Unknown"
        if "Bias Type:" in response:
            bias_type = response.split("Bias Type:")[1].split("\n")[0].strip()
        elif "Type:" in response:
            bias_type = response.split("Type:")[1].split("\n")[0].strip()
            
        explanation = "No explanation available"
        if "Explanation:" in response:
            explanation = response.split("Explanation:")[1].strip()
        elif "Summary:" in response:
            explanation = response.split("Summary:")[1].strip()

        return bias_label, bias_score, bias_type, explanation

    except Exception as e:
        print(f"Error in bias detection: {str(e)}")
        return "Error", 0.0, "Unknown", "No explanation available"