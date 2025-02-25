import os
import json
import re
import pandas as pd
from jinja2 import Template
import ollama

DATA_FOLDER = "../data"
CSV_OUTPUT_PATH = "mistral_fr_output.csv"

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
        result = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

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
