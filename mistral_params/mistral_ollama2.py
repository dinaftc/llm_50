import os
import json
import re
import pandas as pd
from jinja2 import Template
import ollama

DATA_FOLDER = "../data"
CSV_OUTPUT_PATH = "mistral2_output.csv"

def extract_people_count(text, filename):
    """Extracts the number of people in a ski outing using Llama via Ollama."""

    prompt_template = Template("""
    Extract **only** the number of people present in a ski outing or event from the given text. 
    Ignore numbers related to **altitude, distance, temperature, or any non-human count**.

    ### **Rules:**
    1. Extract **only** numbers indicating the **presence of people**.
    2. Ignore mentions of **altitude, distances, speed, weather, or any unrelated numerical values**.
    3. **Ignore numbers referring to people leaving, quitting, or departing from the event.**
    4. If a phrase mentions a **total number of participants**, use that number.
    5. If multiple numbers appear in a sequence, **sum them up**.
    6. If a writer mentions **themselves and at least one other person**, assume a minimum of **2**.
    - Example: "I went skiing with a friend" → Count as **2**.
    - Example: "I went skiing with John and Ricardo" → Count as **3**.
    - Example: "I was there with my group" → If no specific number is given, assume **3**.
    7. If a **group of unnamed people** is mentioned (e.g., "un peu de monde", "quelques personnes"), assume **3-4 people**.
    8. If **no valid numbers** are found, but text exists, assume **the writer is present** and if there are people's names mentioned, count them as well; otherwise, if only the writer is present, return `{filename}: 1`.
    9. **Return ONLY a valid JSON object, with no extra text, explanations, or comments.**

    Now, process the following ski outing description and return the extracted numbers in **valid JSON format**:

    Text:
    {{ text }}

    Return **ONLY** this JSON **with no extra text**:
    ```json
    {
        "filename": "{{ filename }}",
        "number_of_people": ___
    }
    ```
    """)

    prompt = prompt_template.render(text=text, filename=filename)

    try:
        result = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}],options={
        "temperature": 0.7,  # Adjust creativity
        "top_k": 50,         # Limit token sampling
        "top_p": 0.85,       # Adjust randomness
        "repeat_penalty": 1.1  # Prevent repetition
    })

        if not result or "message" not in result or "content" not in result["message"]:
            raise ValueError("No valid response from LLM.")

        output_text = result["message"]["content"].strip()

        # Extract JSON from response using regex
        json_match = re.search(r"\{[\s\S]*?\}", output_text)
        if json_match:
            json_str = json_match.group(0).strip()
        else:
            raise ValueError("No valid JSON found in response.")

        # Parse JSON
        output_json = json.loads(json_str)

        # Ensure valid structure
        if not isinstance(output_json, dict) or "number_of_people" not in output_json:
            raise ValueError("Invalid JSON format.")

        save_model_output({
            "filename": filename,
            "number_of_people": output_json["number_of_people"]
        })

        return {
            "filename": filename,
            "number_of_people": output_json["number_of_people"]
        }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing file {filename}: {e}")
        return {"filename": filename, "number_of_people": 0}

def process_txt_files():
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
    """
    Takes a JSON output from the model and appends it to a CSV file.
    If the file doesn't exist, it creates one.
    """
    try:
        df = pd.DataFrame([output_json])

        # Append to CSV file, create header only if file does not exist
        df.to_csv(CSV_OUTPUT_PATH, mode="a", index=False, header=not os.path.exists(CSV_OUTPUT_PATH))

        print(f"Saved model output to {CSV_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving model output: {e}")

if __name__ == "__main__":
    process_txt_files()
