# Mallikarjuna - GeminiAI LLM Call

import os
import google.generativeai as genai

def generate_answer(
    prompt: str,
    api_key: str = None,
    model_name: str = "gemini-1.5-flash",  # Model name for Gemini 1.5 Flash
    temperature: float = 0.2,
) -> str:
    """
    Generate an answer from Gemini 1.5 Flash using the provided prompt.
    Args:
        prompt (str): The formatted prompt string (context + user question).
        model_name (str): Gemini model code (default: "gemini-1.5-flash").
        temperature (float): Controls randomness of the output (0 = deterministic).
        api_key (str, optional): Gemini API key. If None, reads from GEMINI_API_KEY or GOOGLE_API_KEY env var.
    Returns:
        str: The generated answer from the LLM.
    """

    # Step 1: Get the API key from argument or environment variable
    if api_key is None:
        # Try both common variable names for compatibility
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")

    # Step 2: Configure the Gemini client with your API key
    genai.configure(api_key=api_key)

    # Step 3: Create a Gemini GenerativeModel object with the desired model and parameters
    model = genai.GenerativeModel(model_name=model_name)

    # Step 4: Send the prompt to Gemini and get the response
    response = model.generate_content(prompt)

    # Step 5: Return the generated text as a string
    return response.text