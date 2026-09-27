import json
import re
from datetime import datetime
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

# ---------------------------------------------------------------------------
# Prompt : extraction du contrat (page 3 du PDF)
# ---------------------------------------------------------------------------

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
- is_active must be a boolean: true if the policy is currently active (today), otherwise false (set to null if you cannot determine).

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

# ---------------------------------------------------------------------------
# Prompt : comparaison de la cause du sinistre aux causes couvertes (LLM)
# ---------------------------------------------------------------------------

cause_check_prompt_template = """
You are an insurance coverage expert. Your task is to determine whether a reported incident cause is covered by an insurance policy.

You will be given:
1. The declared cause of the incident (from the FNOL - First Notice of Loss)
2. A list of covered causes from the insurance policy
3. A list of excluded causes from the insurance policy

Your job is to reason carefully and determine:
- Is the declared cause semantically covered by one of the covered causes?
- Is the declared cause semantically excluded by one of the excluded causes?

Rules:
- Be semantic: the wording may differ but the meaning could match (e.g. "pilot's failure to maintain directional control" matches "Pilot Mistakes or failures").
- If the cause matches an excluded cause, it is NOT covered even if it also matches a covered cause.
- Return ONLY a JSON object, no markdown fences, no explanation outside JSON.

JSON schema:
{{
  "cause_is_covered": true | false,
  "matched_covered_cause": "<the matching covered cause string, or null>",
  "matched_excluded_cause": "<the matching excluded cause string, or null>",
  "reasoning": "<one sentence explaining the decision>"
}}

Declared incident cause:
{incident_cause}

Covered causes:
{covered_causes}

Excluded causes:
{excluded_causes}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
                temperature=0.0,
            ),
        )
        json_str = response.text
    except Exception as e:
        print(f"❌ Erreur API Gemini : {e}")
        return {}

    json_str = re.sub(r"```json\s*|\s*```", "", json_str).strip()

    try:
        extracted = json.loads(json_str)
        return _normalize_nan(extracted)
    except json.JSONDecodeError:
        print("❌ JSON invalide reçu de Gemini")
        print("Réponse brute :", json_str[:300])
        return {}


def _parse_date(date_str: str) -> datetime | None:
    """Tente de parser une date en DD/MM/YYYY ou YYYY-MM-DD."""
    if not date_str:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def extraire_contrat(texte: str) -> dict:
    """
    Extrait les champs de la police d'assurance (page 3 du PDF).
    Retourne un dictionnaire Python avec :
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


def verifier_couverture(fnol_data: dict, contrat_data: dict) -> dict:
    """
    Compare les données du sinistre (FNOL) à la police d'assurance (contrat).

    Vérifications effectuées :
      1. La police est-elle active à la date du sinistre ?
      2. La catégorie d'aéronef est-elle couverte ?
      3. La cause du sinistre est-elle couverte (via LLM) ?

    Paramètres
    ----------
    fnol_data    : dict — sortie de extraire_un_fnol()
    contrat_data : dict — sortie de extraire_contrat()

    Retourne
    --------
    dict avec :
        - is_covered  : bool | None
        - reason      : str  (explication lisible)
        - details     : dict (détail de chaque vérification)
    """
    details = {}
    reasons = []

    # ------------------------------------------------------------------
    # 1. Vérification de la période de validité
    # ------------------------------------------------------------------
    event_date_str  = fnol_data.get("EventDate") or ""
    active_from_str = contrat_data.get("active_from") or ""
    active_to_str   = contrat_data.get("active_to") or ""

    event_date  = _parse_date(event_date_str)
    active_from = _parse_date(active_from_str)
    active_to   = _parse_date(active_to_str)

    if event_date and active_from and active_to:
        date_ok = active_from <= event_date <= active_to
        details["date_check"] = {
            "passed": date_ok,
            "event_date": event_date_str,
            "active_from": active_from_str,
            "active_to": active_to_str,
        }
        if not date_ok:
            reasons.append(
                f"La date du sinistre ({event_date_str}) est hors de la période de validité "
                f"de la police ({active_from_str} → {active_to_str})."
            )
    else:
        details["date_check"] = {"passed": None, "note": "Dates manquantes, vérification impossible."}

    # ------------------------------------------------------------------
    # 2. Vérification de la catégorie d'aéronef
    # ------------------------------------------------------------------
    aircraft_category   = (fnol_data.get("AirCraftCategory") or "").strip().upper()
    covered_aircrafts   = [c.strip().upper() for c in (contrat_data.get("covered_aircrafts") or [])]

    if aircraft_category and covered_aircrafts:
        category_ok = aircraft_category in covered_aircrafts
        details["category_check"] = {
            "passed": category_ok,
            "aircraft_category": aircraft_category,
            "covered_aircrafts": covered_aircrafts,
        }
        if not category_ok:
            reasons.append(
                f"La catégorie d'aéronef « {aircraft_category} » n'est pas couverte par la police "
                f"(couverts : {', '.join(covered_aircrafts)})."
            )
    else:
        details["category_check"] = {"passed": None, "note": "Catégorie ou liste manquante."}

    # ------------------------------------------------------------------
    # 3. Vérification de la cause via LLM (Gemini)
    # ------------------------------------------------------------------
    incident_cause  = fnol_data.get("ProbableCause") or ""
    covered_causes  = contrat_data.get("covered_causes") or []
    excluded_causes = contrat_data.get("excluded_causes") or []

    cause_result = {}
    if incident_cause and (covered_causes or excluded_causes):
        prompt = cause_check_prompt_template.format(
            incident_cause=incident_cause,
            covered_causes="\n".join(f"- {c}" for c in covered_causes),
            excluded_causes="\n".join(f"- {c}" for c in excluded_causes),
        )
        cause_result = _call_gemini(prompt)
        cause_ok = cause_result.get("cause_is_covered")
        details["cause_check"] = {
            "passed": cause_ok,
            "incident_cause": incident_cause,
            "matched_covered_cause": cause_result.get("matched_covered_cause"),
            "matched_excluded_cause": cause_result.get("matched_excluded_cause"),
            "reasoning": cause_result.get("reasoning"),
        }
        if cause_ok is False:
            reasons.append(
                f"La cause du sinistre n'est pas couverte : {cause_result.get('reasoning', '')}"
            )
    else:
        details["cause_check"] = {"passed": None, "note": "Cause ou listes manquantes."}

    # ------------------------------------------------------------------
    # 4. Décision finale
    # ------------------------------------------------------------------
    check_results = [
        details.get("date_check", {}).get("passed"),
        details.get("category_check", {}).get("passed"),
        details.get("cause_check", {}).get("passed"),
    ]

    # Si toutes les vérifications disponibles sont True → couvert
    available = [r for r in check_results if r is not None]
    if not available:
        is_covered = None
        final_reason = "Impossible de déterminer la couverture : données insuffisantes."
    elif all(available):
        is_covered = True
        final_reason = "Le sinistre est couvert : la date, la catégorie d'aéronef et la cause sont toutes conformes à la police."
    else:
        is_covered = False
        final_reason = " | ".join(reasons) if reasons else "Sinistre non couvert."

    return {
        "is_covered": is_covered,
        "reason":     final_reason,
        "details":    details,
    }