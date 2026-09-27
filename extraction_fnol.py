import json
import re
from google import genai
from google.genai import types

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Clé API manquante : ajoute GEMINI_API_KEY dans un fichier .env (voir .env.example)")
MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=API_KEY)

prompt_template = """
You are a highly precise data extraction assistant. Your task is to extract information from the following First Notice of Loss (FNOL) document and convert it into a valid JSON object.
Map the extracted data exactly to this JSON schema.

Important rules:
- If a field value is "nan", empty, or not found, set it to null (JSON null, NOT the string "nan").
- Latitude and Longitude are the two numbers before the city name in "Incident Location". Parse them as floats even if separated by ", " (with a space).
- Dates must stay in ISO 8601 format exactly as they appear.
- AmateurBuilt must be a boolean (true/false), not a string.
- NumberOfEngines, SeriousInjuryCount, MinorInjuryCount must be integers.
- Latitude and Longitude must be floats.

--- EXAMPLE ---
INPUT TEXT:
First Notice of Loss
2024-03-30T04:00:00Z
Claim Number: ANC23LA011
Incident Date: 2024-01-02T15:57:00Z
Flight
Operator: nan
Purpose: PERS
Airport of Record: McGahan Industrial
Incident Location:
60.726361,-151.29916, Kenai, Alaska, United States
Aircraft
Make: PIPER
Model: PA-18-150
Category: AIR
Amateur Built: False
Number of Engines: 1
Damage
Cause: The pilot's failure to maintain directional control during landing in flat light conditions,
 resulting in the airplane to nose over.
Damage: Substantial
Condition: VMC
Injuries
Highest Injury Level: Minor
Number of Serious Injuries: 0
Number of Minor Injuries: 1

EXPECTED JSON OUTPUT:
{{
  "NtsbNo": "ANC23LA011",
  "EventDate": "2024-01-02T15:57:00Z",
  "City": "Kenai",
  "State": "Alaska",
  "Country": "United States",
  "OriginalPublishDate": "2024-03-30T04:00:00Z",
  "HighestInjuryLevel": "Minor",
  "SeriousInjuryCount": 0,
  "MinorInjuryCount": 1,
  "ProbableCause": "The pilot's failure to maintain directional control during landing in flat light conditions, resulting in the airplane to nose over.",
  "Latitude": 60.726361,
  "Longitude": -151.29916,
  "Make": "PIPER",
  "Model": "PA-18-150",
  "AirCraftCategory": "AIR",
  "AirportName": "McGahan Industrial",
  "AmateurBuilt": false,
  "NumberOfEngines": 1,
  "PurposeOfFlight": "PERS",
  "AirCraftDamage": "Substantial",
  "WeatherCondition": "VMC",
  "Operator": null
}}
--- END OF EXAMPLE ---

Now, extract the data from the following text and return ONLY the JSON, no markdown fences:

INPUT TEXT:
{fnol_text}
"""

contrat_prompt_template = """
You are a highly precise data extraction assistant. Your task is to extract information from the following aircraft insurance policy document and convert it into a valid JSON object.

Important rules:
- If a field value is "nan", empty, or not found, set it to null (JSON null, NOT the string "nan").
- policy_number must be an integer if it is a plain number, otherwise a string.
- active_from and active_to must stay in the format found in the document (DD/MM/YYYY or ISO 8601).
- covered_aircrafts must be a list of strings (aircraft category codes: AIR, HELI, GLI, GYRO).
  Map the plain English names to their codes: Airplanes -> AIR, Helicopters -> HELI, Gliders -> GLI, Gyrocopters -> GYRO.
- covered_causes must be a list of strings, one entry per covered cause exactly as written.
- excluded_causes must be a list of strings, one entry per excluded cause exactly as written.
- is_active must be a boolean: true if the policy period covers the incident, otherwise false (set to null if you cannot determine).

--- EXAMPLE ---
INPUT TEXT:
Airplance Insurance Policy
Policy Number 29
Active: From 01/01/2024 to 31/12/2024
Policy Holder: Willow Creek Aviation
For Purpose: BUS
Airport of Record: nan
Aircraft
Make & Model: ROBINSON HELICOPTER R44 II
Category: HELI
Coverage:
Covered aircrafts:
Airplanes
Gliders
Helicopters
Gyrocopters
Excluded:
Anything else
Covered:
Pilot Mistakes or failures
Collision with animals, trees or environment
Mechanical due to engine issues, oil starvation, fuel exhaust
Student or instructor mistakes or failures
Unregistered plane or uncertified pilot
Excluded:
Anything else

EXPECTED JSON OUTPUT:
{{
  "policy_number": 29,
  "policy_holder": "Willow Creek Aviation",
  "active_from": "01/01/2024",
  "active_to": "31/12/2024",
  "purpose": "BUS",
  "airport_of_record": null,
  "insured_make": "ROBINSON HELICOPTER",
  "insured_model": "R44 II",
  "insured_category": "HELI",
  "covered_aircrafts": ["AIR", "GLI", "HELI", "GYRO"],
  "covered_causes": [
    "Pilot Mistakes or failures",
    "Collision with animals, trees or environment",
    "Mechanical due to engine issues, oil starvation, fuel exhaust",
    "Student or instructor mistakes or failures",
    "Unregistered plane or uncertified pilot"
  ],
  "excluded_causes": ["Anything else"]
}}
--- END OF EXAMPLE ---

Now, extract the data from the following text and return ONLY the JSON, no markdown fences:

INPUT TEXT:
{contrat_text}
"""


def _normalize_nan(obj):
    """Remplace récursivement la chaîne 'nan' par None après parsing JSON."""
    if isinstance(obj, dict):
        return {k: _normalize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_nan(i) for i in obj]
    if obj == "nan":
        return None
    return obj


def _call_gemini(prompt: str) -> dict:
    """Appelle Gemini et retourne un dict Python. Retourne {} en cas d'erreur."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        json_str = response.text
    except Exception as e:
        print(f"❌ Erreur API Gemini : {e}")
        return {}

    json_str = re.sub(r"```json\s*|\s*```", "", json_str).strip()

    try:
        extracted_data = json.loads(json_str)
        return _normalize_nan(extracted_data)
    except json.JSONDecodeError:
        print("❌ JSON invalide reçu de Gemini")
        print("Réponse brute :", json_str[:300])
        return {}


def extraire_un_fnol(texte: str) -> dict:
    """
    Extrait les champs du FNOL (page 1 du PDF).
    Prend le texte brut et renvoie un dictionnaire Python.
    """
    if not texte or not texte.strip():
        return {}
    return _call_gemini(prompt_template.format(fnol_text=texte))


def extraire_contrat(texte: str) -> dict:
    """
    Extrait les champs du contrat d'assurance (page 3 du PDF).
    Prend le texte brut et renvoie un dictionnaire Python avec :
      - policy_number, policy_holder, active_from, active_to
      - purpose, airport_of_record
      - insured_make, insured_model, insured_category
      - covered_aircrafts  (liste de codes : AIR, HELI, GLI, GYRO)
      - covered_causes     (liste de chaînes)
      - excluded_causes    (liste de chaînes)
    """
    if not texte or not texte.strip():
        return {}
    return _call_gemini(contrat_prompt_template.format(contrat_text=texte))