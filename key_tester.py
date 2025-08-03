import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_api_key():
    """
    This script tests if your Gemini API key is configured correctly in your .env file.
    """
    print("--- Starting API Key Test ---")
    
    try:
        # 1. Load the .env file
        print("Attempting to load .env file...")
        load_dotenv()
        
        # 2. Read the API Key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("\n[FAIL] Could not find GEMINI_API_KEY in your .env file.")
            print("Please ensure your .env file exists and contains the line: GEMINI_API_KEY='AIza...'")
            return

        print("Found API key in .env file.")
        
        # 3. Configure the Gemini client
        genai.configure(api_key=api_key)
        print("Configured the Gemini client.")
        
        # 4. Make a simple test call to the API
        print("Making a test call to the Gemini API...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hello.")
        
        if response.text:
            print("\n[SUCCESS!] Your API Key is working correctly.")
            print(f"Received response from AI: '{response.text.strip()}'")
            print("\nYou can now run your main application.")
        else:
            print("\n[FAIL] The API call succeeded but returned no text. There might be an issue with the model or your account permissions.")

    except Exception as e:
        print("\n--- TEST FAILED ---")
        print("An error occurred. This usually means your API key is invalid or your Google Cloud project is not configured correctly.")
        print(f"\nError Details: {e}\n")
        print("Please follow the checklist: 1. Create a new API key. 2. Ensure the 'Generative Language API' is enabled. 3. Ensure billing is active on your project.")

if __name__ == "__main__":
    test_api_key()