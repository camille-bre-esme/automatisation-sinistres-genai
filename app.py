import gradio as gr
import fitz
import json
import os
from PIL import Image
import io
import base64

from extraction_fnol import extraire_un_fnol
from vision import plausibility_check
from coverage import extraire_contrat, verifier_couverture

_current_pdf_path = None

FIELD_PAGE = {
    "NtsbNo":             0,
    "EventDate":          0,
    "EventTime":          0,
    "Make":               0,
    "Model":              0,
    "City":               0,
    "ProbableCause":      0,
    "AirCraftCategory":   0,
    "AirCraftDamage":     0,
    "HighestInjuryLevel": 0,
    "policy_number":      2,
    "active_from":        2,
    "covered_aircrafts":  2,
    "covered_causes":     2,
}


def _render_page_with_highlight(pdf_path: str, page_idx: int, search_text: str) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        return "<p style='color:#888'>Aucun PDF chargé.</p>"
    try:
        pdf = fitz.open(pdf_path)
        if page_idx >= len(pdf):
            pdf.close()
            return f"<p style='color:#c00'>Page {page_idx + 1} introuvable dans le PDF.</p>"

        page = pdf[page_idx]
        search_text = str(search_text).strip()
        hits = []
        if search_text:
            hits = page.search_for(search_text)
            if not hits and len(search_text) > 20:
                hits = page.search_for(search_text[:40])

        for rect in hits:
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=[1, 0.9, 0.0])
            annot.update()

        mat    = fitz.Matrix(2.0, 2.0)
        pixmap = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pixmap.tobytes("png")
        pdf.close()

        b64 = base64.b64encode(png_bytes).decode()

        found_msg = (
            f"<p style='color:#4ade80;font-size:0.85em;margin:4px 0'>{len(hits)} occurrence(s) surlignée(s)</p>"
            if hits else
            "<p style='color:#f59e0b;font-size:0.85em;margin:4px 0'>Texte introuvable sur cette page — page affichée sans surlignage</p>"
        )

        scroll_script = """
        <script>
            (function() {
                function scrollToViewer(doc) {
                    let el = doc.getElementById('viewer');
                    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); return true; }

                    el = doc.querySelector('[id$="pdf_viewer"]');
                    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); return true; }

                    let headings = doc.querySelectorAll('h2, h3, p, span');
                    for (let h of headings) {
                        if (h.textContent.includes('Localisation')) {
                            h.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            return true;
                        }
                    }
                    return false;
                }

                setTimeout(function() {
                    try {
                        if (scrollToViewer(document)) return;

                        let win = window;
                        while (win !== win.parent) {
                            win = win.parent;
                            try {
                                if (scrollToViewer(win.document)) return;
                            } catch(e) {}
                        }
                    } catch(e) {}
                }, 200);
            })();
        </script>
        """
        return (
            scroll_script +
            found_msg +
            f'<img id="pdf-highlight-img" src="data:image/png;base64,{b64}" '
            f'style="width:100%;border:1px solid #334155;border-radius:8px;margin-top:4px" />'
        )
    except Exception as e:
        return f"<p style='color:#f87171'>Erreur lors du rendu PDF : {e}</p>"


def analyser_sinistre(pdf_file):
    global _current_pdf_path

    empty_viewer = "<p style='color:#64748b;font-family:monospace;padding:12px'>Cliquez sur une loupe pour localiser le champ dans le PDF.</p>"

    if pdf_file is None:
        _current_pdf_path = None
        return ([""] * 14 + [None, "", "", "", empty_viewer, "{}"])

    _current_pdf_path = pdf_file
    pdf = fitz.open(pdf_file)

    texte_fnol    = pdf[0].get_text() if len(pdf) >= 1 else ""
    texte_contrat = pdf[2].get_text() if len(pdf) >= 3 else ""

    data    = extraire_un_fnol(texte_fnol)
    contrat = extraire_contrat(texte_contrat)

    if data.get("EventDate") and "T" in data.get("EventDate", ""):
        parts = data["EventDate"].split("T")
        data["EventDate"] = parts[0]
        data["EventTime"] = parts[1].replace("Z", "")

    pil_image = None
    if len(pdf) >= 2:
        pil_image = Image.open(io.BytesIO(pdf[1].get_pixmap(dpi=150).tobytes("png")))

    pdf.close()

    result_p     = plausibility_check(pil_image, data.get("AirCraftCategory", ""))
    plausibilite = "PLAUSIBLE" if result_p["is_plausible"] else "NON PLAUSIBLE"
    confidence   = f"{result_p['confidence']}%"
    predicted    = result_p["predicted_category"]

    coverage = verifier_couverture(data, contrat)

    is_covered_str = (
        "COUVERT"          if coverage.get("is_covered") is True
        else "NON COUVERT" if coverage.get("is_covered") is False
        else "INDÉTERMINÉ"
    )

    active_from_str     = contrat.get("active_from") or ""
    active_to_str       = contrat.get("active_to") or ""
    contract_period_str = f"{active_from_str} → {active_to_str}" if (active_from_str or active_to_str) else ""

    covered_aircrafts_str = ", ".join(contrat.get("covered_aircrafts") or [])
    covered_causes_str    = "\n".join(f"• {c}" for c in (contrat.get("covered_causes") or []))

    extracted = {
        "NtsbNo":             data.get("NtsbNo", ""),
        "EventDate":          data.get("EventDate", ""),
        "EventTime":          data.get("EventTime", ""),
        "Make":               data.get("Make", ""),
        "Model":              data.get("Model", ""),
        "City":               data.get("City", ""),
        "ProbableCause":      data.get("ProbableCause", ""),
        "AirCraftCategory":   data.get("AirCraftCategory", ""),
        "AirCraftDamage":     data.get("AirCraftDamage", ""),
        "HighestInjuryLevel": data.get("HighestInjuryLevel", ""),
        "policy_number":      contrat.get("policy_number", ""),
        "active_from":        active_from_str,
        "covered_aircrafts":  covered_aircrafts_str,
        "covered_causes":     covered_causes_str,
    }

    photo_small = None
    if pil_image:
        photo_small = pil_image.resize(
            (pil_image.width // 2, pil_image.height // 2), Image.LANCZOS
        )

    return [
        extracted["NtsbNo"],
        extracted["EventDate"],
        extracted["EventTime"],
        extracted["Make"],
        extracted["Model"],
        extracted["City"],
        extracted["ProbableCause"],
        extracted["AirCraftCategory"],
        extracted["AirCraftDamage"],
        extracted["HighestInjuryLevel"],
        extracted["policy_number"],
        contract_period_str,
        covered_aircrafts_str,
        covered_causes_str,
        photo_small,
        f"{plausibilite} — {predicted} ({confidence})",
        is_covered_str,
        coverage.get("reason", ""),
        empty_viewer,
        json.dumps(extracted),
    ]


AIRCRAFT_CODE_TO_PDF = {
    "AIR":  "Airplanes",
    "HELI": "Helicopters",
    "GLI":  "Gliders",
    "GYRO": "Gyrocopters",
}


def chercher_dans_pdf(field_key: str, extracted_json: str):
    global _current_pdf_path
    if not _current_pdf_path or not extracted_json:
        return "<p style='color:#f87171'>Analysez d'abord un PDF.</p>"
    try:
        extracted = json.loads(extracted_json)
    except Exception:
        return "<p style='color:#f87171'>Erreur interne (JSON invalide).</p>"

    value = extracted.get(field_key)
    if value is None or str(value).strip() == "":
        return "<p style='color:#64748b'>Valeur vide, rien à surligner.</p>"

    page_idx    = FIELD_PAGE.get(field_key, 0)
    search_text = str(value).strip()

    if field_key == "covered_aircrafts":
        first_code  = search_text.split(",")[0].strip().upper()
        search_text = AIRCRAFT_CODE_TO_PDF.get(first_code, search_text)
    elif field_key == "active_from":
        search_text = f"From {search_text}"
    elif field_key == "covered_causes":
        lines       = [l.lstrip("• ").strip() for l in search_text.split("\n") if l.strip()]
        search_text = lines[0] if lines else search_text
        search_text = search_text[:40]
    else:
        if "\n" in search_text:
            search_text = search_text.split("\n")[0].lstrip("• ").strip()

    return _render_page_with_highlight(_current_pdf_path, page_idx, search_text)



FNOL_EDIT_KEYS = [
    "NtsbNo", "EventDate", "EventTime", "Make", "Model", "City",
    "ProbableCause", "AirCraftCategory", "AirCraftDamage", "HighestInjuryLevel",
]

def activer_edition():
    """Passe tous les champs FNOL en mode interactif et affiche le bouton Sauvegarder."""
    updates = [gr.update(interactive=True) for _ in FNOL_EDIT_KEYS]
    return updates + [gr.update(visible=False), gr.update(visible=True)]

def sauvegarder_edition(extracted_json, *values):
    """Verrouille les champs, met à jour le state JSON avec les nouvelles valeurs."""
    try:
        extracted = json.loads(extracted_json)
    except Exception:
        extracted = {}

    for key, val in zip(FNOL_EDIT_KEYS, values):
        extracted[key] = val

    updates = [gr.update(interactive=False) for _ in FNOL_EDIT_KEYS]
    return updates + [gr.update(visible=True), gr.update(visible=False), json.dumps(extracted)]



FNOL_FIELDS = [
    ("NtsbNo",             "Numéro de sinistre",  1),
    ("EventDate",          "Date du sinistre",    1),
    ("EventTime",          "Heure du sinistre",   1),
    ("Make",               "Marque",              1),
    ("Model",              "Modèle",              1),
    ("City",               "Ville",               1),
    ("ProbableCause",      "Cause probable",      4),
    ("AirCraftCategory",   "Catégorie",           1),
    ("AirCraftDamage",     "Dommages",            1),
    ("HighestInjuryLevel", "Blessures",           1),
    ("policy_number",      "N° Police",           1),
]

col1_keys = ["NtsbNo", "EventDate", "EventTime", "Make", "Model", "City"]
col2_keys = ["ProbableCause", "AirCraftCategory", "AirCraftDamage", "HighestInjuryLevel", "policy_number"]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
}

.sidebar {
    background: #1e293b !important;
    border-right: 1px solid #334155 !important;
}

h1 {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.02em !important;
    border-bottom: 2px solid #3b82f6 !important;
    padding-bottom: 10px !important;
    margin-bottom: 20px !important;
}

h2 {
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-top: 28px !important;
    margin-bottom: 12px !important;
}

.gr-group {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.gr-textbox textarea, .gr-textbox input {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
}

label span {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.gr-button-primary {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: background 0.2s ease !important;
}
.gr-button-primary:hover {
    background: #2563eb !important;
}

.gr-button-secondary {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
}
.gr-button-secondary:hover {
    background: #334155 !important;
    color: #e2e8f0 !important;
}

.gr-file {
    background: #1e293b !important;
    border: 1px dashed #334155 !important;
    border-radius: 10px !important;
}

.gr-image {
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    overflow: hidden !important;
}

.gr-html {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

.gr-row {
    gap: 12px !important;
}

.fnol-editing textarea, .fnol-editing input {
    border: 1px solid #3b82f6 !important;
    background: #1e3a5f !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(), title="Gestion des Sinistres", css=custom_css) as demo:

    extracted_state = gr.State("{}")

    with gr.Sidebar(open=True):
        gr.Markdown("## Navigation")
        gr.Markdown("""
- [Depot du PDF](#depot)
- [Resultat de l'analyse](#couverture)
- [Informations FNOL](#fnol)
- [Contrat d'assurance](#contrat)
- [Analyse visuelle](#visuel)
- [Localisation PDF](#viewer)
        """)

    gr.Markdown("# Gestion des Sinistres Aéronautiques", elem_id="depot")

    with gr.Row():
        pdf_input = gr.File(label="Déposer un sinistre PDF", file_types=[".pdf"])
        btn       = gr.Button("Analyser", variant="primary", scale=0, min_width=160)

    gr.Markdown("## Résultat de l'analyse", elem_id="couverture")
    with gr.Row():
        covered_out      = gr.Textbox(label="Statut de couverture", interactive=False)
        plausibilite_out = gr.Textbox(label="Plausibilité", lines=2, interactive=False)
        reason_out       = gr.Textbox(label="Motif", lines=3, interactive=False)

    gr.Markdown("## Informations extraites du FNOL", elem_id="fnol")
    with gr.Row():
        btn_edit = gr.Button("✏️ Modifier", variant="secondary", scale=0, min_width=140)
        btn_save = gr.Button("💾 Sauvegarder", variant="primary", scale=0, min_width=140, visible=False)

    textbox_components = {}
    loupe_buttons      = {}

    with gr.Row():
        with gr.Column():
            for key, label, lines in FNOL_FIELDS:
                if key not in col1_keys:
                    continue
                with gr.Group():
                    with gr.Row(equal_height=True):
                        tb = gr.Textbox(label=label, lines=lines, scale=9, interactive=False)
                        lb = gr.Button("🔍", scale=1, min_width=40)
                    textbox_components[key] = tb
                    loupe_buttons[key]      = lb

        with gr.Column():
            for key, label, lines in FNOL_FIELDS:
                if key not in col2_keys:
                    continue
                with gr.Group():
                    with gr.Row(equal_height=True):
                        tb = gr.Textbox(label=label, lines=lines, scale=9, interactive=False)
                        lb = gr.Button("🔍", scale=1, min_width=40)
                    textbox_components[key] = tb
                    loupe_buttons[key]      = lb

    gr.Markdown("## Informations du contrat d'assurance", elem_id="contrat")

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row(equal_height=True):
                    contract_period_tb = gr.Textbox(
                        label="Période de validité", lines=1, scale=9, interactive=False
                    )
                    lb_period = gr.Button("🔍", scale=1, min_width=40)

            with gr.Group():
                with gr.Row(equal_height=True):
                    covered_vehicles_tb = gr.Textbox(
                        label="Véhicules couverts", lines=1, scale=9, interactive=False
                    )
                    lb_vehicles = gr.Button("🔍", scale=1, min_width=40)

        with gr.Column():
            with gr.Group():
                with gr.Row(equal_height=True):
                    covered_causes_tb = gr.Textbox(
                        label="Causes couvertes", lines=6, scale=9, interactive=False
                    )
                    lb_causes = gr.Button("🔍", scale=1, min_width=40)

    gr.Markdown("## Analyse visuelle", elem_id="visuel")
    with gr.Row():
        photo_out = gr.Image(label="Photo du sinistre")

    gr.Markdown("## Localisation dans le PDF", elem_id="viewer")
    pdf_viewer = gr.HTML(
        value="<p style='color:#64748b;font-family:monospace;padding:8px'>Cliquez sur la loupe pour localiser le champ dans le PDF</p>",
        elem_id="pdf-viewer-html"
    )

    _scroll_js = """
    () => {
        setTimeout(function() {
            let el = document.getElementById('viewer');
            if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); return; }
            el = document.getElementById('pdf-viewer-html');
            if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); return; }
            for (let h of document.querySelectorAll('h2, h3, span, div')) {
                if (h.textContent.trim().startsWith('Localisation')) {
                    h.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    return;
                }
            }
        }, 300);
    }
    """

    outputs_list = (
        [textbox_components[k] for k in
         ["NtsbNo", "EventDate", "EventTime", "Make", "Model", "City",
          "ProbableCause", "AirCraftCategory", "AirCraftDamage", "HighestInjuryLevel", "policy_number"]]
        + [contract_period_tb, covered_vehicles_tb, covered_causes_tb]
        + [photo_out, plausibilite_out, covered_out, reason_out, pdf_viewer, extracted_state]
    )

    btn.click(fn=analyser_sinistre, inputs=[pdf_input], outputs=outputs_list)

    fnol_textboxes = [textbox_components[k] for k in FNOL_EDIT_KEYS]

    btn_edit.click(
        fn=activer_edition,
        inputs=[],
        outputs=fnol_textboxes + [btn_edit, btn_save],
    )

    btn_save.click(
        fn=sauvegarder_edition,
        inputs=[extracted_state] + fnol_textboxes,
        outputs=fnol_textboxes + [btn_edit, btn_save, extracted_state],
    )

    for key, btn_loupe in loupe_buttons.items():
        btn_loupe.click(
            fn=lambda st, k=key: chercher_dans_pdf(k, st),
            inputs=[extracted_state],
            outputs=[pdf_viewer],
            js=_scroll_js,
        )

    lb_period.click(
        fn=lambda st: chercher_dans_pdf("active_from", st),
        inputs=[extracted_state],
        outputs=[pdf_viewer],
        js=_scroll_js,
    )
    lb_vehicles.click(
        fn=lambda st: chercher_dans_pdf("covered_aircrafts", st),
        inputs=[extracted_state],
        outputs=[pdf_viewer],
        js=_scroll_js,
    )
    lb_causes.click(
        fn=lambda st: chercher_dans_pdf("covered_causes", st),
        inputs=[extracted_state],
        outputs=[pdf_viewer],
        js=_scroll_js,
    )


demo.launch()