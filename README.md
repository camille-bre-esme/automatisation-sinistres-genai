# ✈️ Automatisation de sinistres avec l'IA générative

Outil qui automatise le traitement d'une déclaration de sinistre aéronautique : à partir d'un simple PDF, il extrait les informations clés, vérifie que la photo correspond au véhicule déclaré et détermine si le sinistre est couvert par le contrat d'assurance.

> Projet réalisé dans le cadre de ma formation d'ingénieure IA & Data à l'ESME Sudria.


https://github.com/user-attachments/assets/e115903a-becc-4f21-a206-6ba88626df13




## 🎯 Fonctionnalités

- **Extraction d'informations (NLP)** : lecture du FNOL (*First Notice of Loss*, la déclaration de sinistre) et extraction structurée en JSON par un LLM, avec du prompting few-shot
- **Vérification de plausibilité (Computer Vision)** : classification zero-shot de la photo du sinistre avec CLIP (avion, hélicoptère, planeur ou autogire), comparée au type d'appareil déclaré
- **Analyse de couverture** : extraction des garanties du contrat, puis vérification combinant règles métier (appareils couverts, dates de validité) et raisonnement sémantique du LLM sur la cause du sinistre
- **Interface web (Gradio)** : dépôt d'un PDF, résultat de l'analyse, correction manuelle des champs extraits et localisation de chaque information directement dans le document source 🔍

## 🧠 Fonctionnement

Chaque sinistre est un PDF de trois pages : la déclaration (FNOL), une photo de l'appareil et le contrat d'assurance.

```
PDF du sinistre
 ├── Page 1 : FNOL     ──► LLM (Gemini) ──► données du sinistre (JSON)
 ├── Page 2 : Photo    ──► CLIP          ──► type d'appareil détecté
 └── Page 3 : Contrat  ──► LLM (Gemini) ──► garanties couvertes
                                    │
                                    ▼
           Plausibilité + couverture ──► décision affichée dans l'interface
```

## 🛠️ Technologies

Python · Google Gemini (`gemini-2.5-flash`) · OpenCLIP (ViT-L/14) · PyTorch · PyMuPDF · Gradio

## 📁 Structure du projet

```
├── app.py               # Interface Gradio et orchestration du pipeline
├── extraction_fnol.py   # Extraction des informations du FNOL par LLM
├── vision.py            # Vérification de plausibilité avec CLIP
├── coverage.py          # Extraction du contrat et vérification de couverture
├── data/exemples/       # Quelques sinistres pour tester l'application
├── requirements.txt
└── .env.example         # Modèle de fichier pour la clé API
```

## 🚀 Installation et lancement

1. Cloner le dépôt et installer les dépendances :

```bash
git clone https://github.com/camille-bre-esme/automatisation-sinistres-genai.git
cd automatisation-sinistres-genai
pip install -r requirements.txt
```

2. Créer un fichier `.env` à partir de `.env.example` et y ajouter sa clé API Gemini (gratuite sur [Google AI Studio](https://aistudio.google.com/apikey)) :

```
GEMINI_API_KEY=ta_cle_api
```

3. Lancer l'application :

```bash
python app.py
```

L'interface s'ouvre dans le navigateur. Il suffit ensuite de déposer un des PDF du dossier `data/exemples/`.

> Au premier lancement, le modèle CLIP (environ 1,7 Go) est téléchargé automatiquement. Un GPU accélère l'analyse mais n'est pas obligatoire.

## 📊 Données

Les sinistres utilisés proviennent du dépôt pédagogique [lemans-courses-share](https://github.com/atracordis/lemans-courses-share), construit à partir de rapports d'accidents aériens publics. Seuls quelques exemples sont inclus ici ; le jeu complet est disponible sur ce dépôt.

## 💡 Pistes d'amélioration

- Évaluer automatiquement les extractions sur l'ensemble du jeu de données
- Tester un LLM open source exécuté en local pour ne plus dépendre d'une API
- Conteneuriser l'application avec Docker pour faciliter son déploiement
