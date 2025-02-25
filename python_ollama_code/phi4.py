import os
import json
import re
import pandas as pd
from jinja2 import Template
import ollama

# Constants
DATA_FOLDER = "../data"
CSV_OUTPUT_PATH = "phi4_output.csv"
MODEL_NAME = "phi4"  # Switch to "phi3:medium" if needed
MAX_INPUT_LENGTH = 1500  # Avoid API truncation issues

def extract_people_count(text, filename):
    """Extracts the number of people in a ski outing using Llama via Ollama."""

    # Truncate text if too long
    truncated_text = text[:MAX_INPUT_LENGTH]

    # Enforce strict JSON output
    prompt_template = Template("""
    Extract the **number of people** present in a ski outing from the given text.
    Return the result **strictly** in JSON format, with **no extra text**.

    ## **Rules:**
    1. **Extract only** numbers indicating **people present**.
    2. Ignore numbers related to **altitude, distance, temperature, speed, weather, or any non-human count**.
    3. Ignore numbers about **people leaving, quitting, or departing**.
    4. If a phrase mentions a **total number of participants**, use that number.
    5. If multiple numbers represent people, **sum them up**.
    6. If **only the writer is present**, return `{ "filename": "{{ filename }}", "number_of_people": 1 }`.
    7. If **no valid number is found but names appear**, count named individuals.
    8. If **a group** is mentioned (e.g., "some people", "a few friends"), assume **3-4 people**.
    9. **Return JSON only**, without explanations.

    ## **Input Text:**
    {{ truncated_text }}

    ## **Expected JSON Output Format:**
    ```json
    {
        "filename": "{{ filename }}",
        "number_of_people": ___
    }
    Return only the JSON object, without extra text. """)
    prompt = prompt_template.render(truncated_text=truncated_text, filename=filename)

    try:
        # Send request to Ollama
        result = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        print(result)

        # Validate response
        if not result or "message" not in result or "content" not in result["message"]:
            raise ValueError("No valid response from LLM.")

        output_text = result["message"]["content"].strip()
        print(output_text)

        # Extract JSON from response using regex
        json_match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", output_text)
        if not json_match:
            raise ValueError("No valid JSON found in response.")

        json_str = json_match.group(0).strip()

        # Parse JSON safely
        output_json = json.loads(json_str)

        # Ensure valid structure
        if not isinstance(output_json, dict) or "number_of_people" not in output_json:
            raise ValueError("Invalid JSON format.")

        # Save and return results
        save_model_output(output_json)
        return output_json

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing file {filename}: {e}")
        return {"filename": filename, "number_of_people": 0}

def process_txt_files(): 
    """Processes all .txt files in the data folder.""" 
    if not os.path.exists(DATA_FOLDER): 
        print(f"Folder {DATA_FOLDER} does not exist.") 
        return
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                print(f"Processing file: {filename}")
                result = extract_people_count(text, filename)
                print(result)


def save_model_output(output_json): 
    """Appends extracted data to a CSV file.""" 
    try: 
        df = pd.DataFrame([output_json])
        # Append to CSV file, create header if file does not exist
        df.to_csv(CSV_OUTPUT_PATH, mode="a", index=False, header=not os.path.exists(CSV_OUTPUT_PATH))

        print(f"Saved model output to {CSV_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving model output: {e}")


if __name__ == "__main__": 
    process_txt_files()
