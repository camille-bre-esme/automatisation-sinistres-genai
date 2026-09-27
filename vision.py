import torch
import open_clip
from PIL import Image

# ---------------------------------------------------------------------------
# Chargement du modèle (une seule fois à l'import du module)
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
_model = _model.to(device)
_tokenizer = open_clip.get_tokenizer("ViT-L-14")

LABEL_MAPPING = {
    "AIR":  "a photo of an airplane",
    "HELI": "a photo of a helicopter",
    "GLI":  "a photo of a glider",
    "GYRO": "a photo of a gyrocopter",
}
REVERSE_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
_labels = list(LABEL_MAPPING.values())

# ---------------------------------------------------------------------------
# Fonction principale — à appeler depuis app.py
# ---------------------------------------------------------------------------

def plausibility_check(pil_image, category_from_fnol):
    """
    Classifie le véhicule présent sur la photo (PIL Image) via open_clip,
    puis compare au type de véhicule déclaré dans le FNOL.

    Paramètres
    ----------
    pil_image           : PIL.Image — image extraite de la page 2 du PDF
    category_from_fnol  : str       — catégorie extraite par extraction_fnol
                                      (ex : "AIR", "HELI", "GLI", "GYRO")

    Retourne
    --------
    dict avec :
        - is_plausible       : bool | None  (None si image absente)
        - predicted_category : str          (catégorie détectée sur la photo)
        - confidence         : float        (score de confiance en %)
    """
    if pil_image is None:
        return {"is_plausible": None, "predicted_category": "N/A", "confidence": 0.0}

    image_tensor = _preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
    text_tokens  = _tokenizer(_labels).to(device)

    with torch.no_grad():
        img_feat = _model.encode_image(image_tensor)
        txt_feat = _model.encode_text(text_tokens)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)

    pred_idx   = probs[0].argmax().item()
    predicted  = REVERSE_MAPPING[_labels[pred_idx]]
    confidence = round(probs[0][pred_idx].item() * 100, 1)

    is_plausible = (predicted == category_from_fnol) if category_from_fnol else None

    return {
        "is_plausible":       is_plausible,
        "predicted_category": predicted,
        "confidence":         confidence,
    }
