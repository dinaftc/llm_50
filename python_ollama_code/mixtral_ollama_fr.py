import os
import json
import re
import pandas as pd
from jinja2 import Template
import ollama

DATA_FOLDER = "../data"
CSV_OUTPUT_PATH = "mixtral_fr_output.csv"

# Extract JSON safely
# Extract JSON safely
def extract_json(text):
    text = text.strip()
    json_start = text.find("{")
    json_end = text.rfind("}")

    if json_start == -1 or json_end == -1:
        raise ValueError("No valid JSON found in response.")

    json_str = text[json_start : json_end + 1]
    json_str = re.sub(r"\\_", "_", json_str)  # Fix invalid escapes
    json_str = re.sub(r"\\n", "", json_str)   # Remove stray newlines if needed

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def extract_people_count(text, filename):
    """Extracts the number of people in a ski outing using Llama via Ollama."""

    prompt_template = Template("""
    Extrayez **uniquement** le nombre de personnes présentes lors d'une sortie ou d'un événement de ski à partir du texte donné.
    Ignorez les nombres liés à **l'altitude, la distance, la température ou tout autre comptage non humain**.

    ### **Règles :**
    1. Extrayez **uniquement** les nombres indiquant la **présence de personnes**.
    2. Ignorez les mentions de **l'altitude, des distances, de la vitesse, de la météo ou de toute valeur numérique non pertinente**.
    3. **Ignorez les nombres faisant référence aux personnes quittant, abandonnant ou partant de l'événement.**
    4. Si une phrase mentionne un **nombre total de participants**, utilisez ce nombre.
    5. Si plusieurs nombres apparaissent en séquence, **sommez-les**.
    6. Si l'auteur mentionne **lui-même et au moins une autre personne**, supposez un minimum de **2**.
    - Exemple : "Je suis allé skier avec un ami" → Comptez **2**.
    - Exemple : "Je suis allé skier avec John et Ricardo" → Comptez **3**.
    - Exemple : "J'étais là avec mon groupe" → Si aucun nombre spécifique n'est donné, supposez **3**.
    - Exemple : "Nous avons pris la route 5" → Comptez **3**.
    7. Si un **groupe de personnes non nommées** est mentionné (ex. : "un peu de monde", "quelques personnes"), supposez **3 personnes**.
    8. Si **aucun nombre valide** n'est trouvé mais que du texte est présent, supposez **que l'auteur est présent** et, si des noms de personnes sont mentionnés, comptez-les également ; sinon, si seul l'auteur est présent, retournez `{filename}: 1`.
    9. **Retournez UNIQUEMENT un objet JSON valide, sans texte supplémentaire, explications ou commentaires.**

    Maintenant, traitez la description suivante de la sortie de ski et retournez le nombre extrait au format **JSON valide** :

    Texte :
    {{ text }}

    Retournez **UNIQUEMENT** cet objet JSON **sans aucun texte supplémentaire** :
    ```json
    {
        "filename": "{{ filename }}",
        "number_of_people": ___
    }
    ```
    """)


    prompt = prompt_template.render(text=text, filename=filename)
    try:
        result = ollama.chat(model="mixtral:8x7b", messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,  # Keep responses precise and avoid randomness.
            "top_k": 20,  # Select from the top 20 most likely words (ensures consistency).
            "top_p": 0.5,  # Limit responses to high-confidence words.
            "repeat_penalty": 1.2  # Avoid redundant or looping responses.
        })
        print(result)

        if not result or "message" not in result or not result["message"].get("content"):
            print(f"Warning: No valid response from LLM for file {filename}.")
            return {"filename": filename, "number_of_people": 0}

        output_text = result["message"]["content"].strip()
        print(output_text)

        output_json = extract_json(output_text)
        print(output_json)

        # Ensure the JSON has a valid number_of_people field
        output_json.setdefault("number_of_people", 0)

        save_model_output(output_json)

        return output_json

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
