import gradio as gr
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Resolve paths relative to this script so app works when started
# from the repository root or another working directory.
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(BASE_DIR / 'columns.pkl', 'rb') as f:
    all_symptoms = [str(symptom).strip().lower() for symptom in pickle.load(f)]

all_symptoms = sorted(set(all_symptoms))

with open(BASE_DIR / 'label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

desc_df = pd.read_csv(BASE_DIR / 'data' / 'symptom_Description.csv')
prec_df = pd.read_csv(BASE_DIR / 'data' / 'symptom_precaution.csv')


def load_description_table(path):
    table = pd.read_csv(path)
    if 'Disease' not in table.columns:
        table = pd.read_csv(path, header=None, names=['Disease', 'Description'])
    table['Disease'] = table['Disease'].astype(str).str.strip()
    table['Disease_key'] = table['Disease'].str.lower()
    return table


def load_precaution_table(path):
    table = pd.read_csv(path)
    if 'Disease' not in table.columns:
        table = pd.read_csv(
            path,
            header=None,
            names=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'],
        )
    table['Disease'] = table['Disease'].astype(str).str.strip()
    table['Disease_key'] = table['Disease'].str.lower()
    return table


desc_df = load_description_table(BASE_DIR / 'data' / 'symptom_Description.csv')
prec_df = load_precaution_table(BASE_DIR / 'data' / 'symptom_precaution.csv')
severity_df = pd.read_csv(BASE_DIR / 'data' / 'Symptom-severity.csv')
severity_map = dict(zip(
    severity_df['Symptom'].astype(str).str.strip().str.lower(),
    severity_df['weight']
))

symptom_choices = [(symptom.replace('_', ' '), symptom) for symptom in all_symptoms]

MENTAL_SYMPTOMS = {
    'anxiety',
    'depression',
    'mood_swings',
    'irritability',
    'lack_of_concentration',
    'restlessness',
}

def normalize_symptoms(selected_symptoms):
    if not selected_symptoms:
        return []

    if isinstance(selected_symptoms, str):
        selected_symptoms = [selected_symptoms]

    normalized = []
    for symptom in selected_symptoms:
        token = str(symptom).strip().lower()
        if token:
            normalized.append(token)
    return normalized


def display_symptom(symptom):
    return str(symptom).replace('_', ' ').title()


def categorize_selected_symptoms(selected_symptoms):
    mental_symptoms = [symptom for symptom in selected_symptoms if symptom in MENTAL_SYMPTOMS]
    physical_symptoms = [symptom for symptom in selected_symptoms if symptom not in MENTAL_SYMPTOMS]
    return mental_symptoms, physical_symptoms


def get_disease_details(disease):
    disease_key = str(disease).strip().lower()

    desc_row = desc_df[desc_df['Disease_key'] == disease_key]
    description = desc_row['Description'].values[0] if not desc_row.empty else "N/A"

    prec_row = prec_df[prec_df['Disease_key'] == disease_key]
    if not prec_row.empty:
        precautions = [
            value
            for value in prec_row.iloc[0][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].dropna().tolist()
            if str(value).strip()
        ]
    else:
        precautions = []

    return description, precautions


def build_collective_summary(top_predictions, mental_symptoms, physical_symptoms):
    if not top_predictions:
        return ""

    top_disease, top_score, _, _ = top_predictions[0]
    other_diseases = [disease for disease, _, _, _ in top_predictions[1:]]

    all_precautions = [precaution for _, _, _, precautions in top_predictions for precaution in precautions]
    common_precautions = []
    seen_precautions = set()
    for precaution in all_precautions:
        normalized = precaution.strip().lower()
        if normalized and normalized not in seen_precautions:
            seen_precautions.add(normalized)
            common_precautions.append(precaution)

    summary_lines = [
        "Collective response:",
        f"The strongest match is {top_disease} at {top_score}%."
    ]

    if mental_symptoms or physical_symptoms:
        summary_lines.append("Category breakdown:")
        if mental_symptoms:
            summary_lines.append(
                "Mental symptoms: " + ", ".join(display_symptom(symptom) for symptom in mental_symptoms) + "."
            )
        if physical_symptoms:
            summary_lines.append(
                "Physical symptoms: " + ", ".join(display_symptom(symptom) for symptom in physical_symptoms) + "."
            )

    if other_diseases:
        summary_lines.append("Other possible conditions: " + ", ".join(other_diseases) + ".")

    if common_precautions:
        summary_lines.append("Shared precaution themes: " + ", ".join(common_precautions[:4]) + ".")

    summary_lines.append("This summary combines the top predicted diseases so you can see both the main result and the likely alternatives.")
    return "\n".join(summary_lines)

def predict(selected_symptoms):
    selected_symptoms = normalize_symptoms(selected_symptoms)

    if not selected_symptoms or len(selected_symptoms) < 3:
        return "Please select at least 3 symptoms for a meaningful prediction."

    known_symptoms = set(all_symptoms)
    matched_symptoms = [symptom for symptom in selected_symptoms if symptom in known_symptoms]

    if not matched_symptoms:
        return "⚠️ None of the selected symptoms were recognized. Please choose symptoms from the dropdown list."

    selected_symptom_set = set(matched_symptoms)
    mental_symptoms, physical_symptoms = categorize_selected_symptoms(matched_symptoms)
    input_vector = [severity_map.get(symptom, 1) if symptom in selected_symptom_set else 0 for symptom in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    proba = model.predict_proba(input_df)[0]
    top3_idx = np.argsort(proba)[-3:][::-1]

    top_predictions = []
    output = "Top predictions:\n\n"
    for rank, idx in enumerate(top3_idx, 1):
        disease = le.classes_[idx]
        confidence = round(proba[idx] * 100, 2)
        description, precautions = get_disease_details(disease)
        precaution_text = ", ".join(precautions) if precautions else "N/A"
        top_predictions.append((disease, confidence, description, precautions))

        output += f"{rank}. {disease} - {confidence}%\n"
        output += f"Description: {description}\n"
        output += f"Precautions: {precaution_text}\n\n"
        output += "---\n"

    output += "\n" + build_collective_summary(top_predictions, mental_symptoms, physical_symptoms) + "\n\n"
    output += "Warning: This tool is for educational purposes only and is not a substitute for professional medical advice."
    return output


APP_CSS = """
:root {
    --page-bg: #07111f;
    --panel-bg: rgba(9, 17, 32, 0.72);
    --panel-border: rgba(148, 163, 184, 0.18);
    --accent: #ff7a18;
    --accent-2: #ffb347;
    --text-main: #f8fafc;
    --text-soft: #cbd5e1;
    --shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
}

body {
    background:
        radial-gradient(circle at top left, rgba(255, 122, 24, 0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 24%),
        linear-gradient(180deg, #08111f 0%, #060b14 100%);
}

.gradio-container {
    font-family: 'Manrope', 'Segoe UI', sans-serif !important;
}

#app-shell {
    max-width: 1240px;
    margin: 0 auto;
}

.hero-card {
    position: relative;
    overflow: hidden;
    padding: 28px;
    border-radius: 28px;
    border: 1px solid var(--panel-border);
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(8, 15, 28, 0.82));
    box-shadow: var(--shadow);
}

.hero-card::after {
    content: '';
    position: absolute;
    inset: -2px;
    background: linear-gradient(135deg, rgba(255, 122, 24, 0.12), transparent 34%, rgba(59, 130, 246, 0.12));
    pointer-events: none;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    margin-bottom: 14px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    color: var(--text-soft);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-title {
    margin: 0;
    color: var(--text-main);
    font-size: clamp(2rem, 3vw, 3.4rem);
    line-height: 1.02;
    font-weight: 800;
}

.hero-subtitle {
    margin-top: 14px;
    max-width: 760px;
    color: var(--text-soft);
    font-size: 1.02rem;
    line-height: 1.65;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 22px;
}

.feature {
    padding: 16px 18px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-main);
}

.feature strong {
    display: block;
    margin-bottom: 4px;
    font-size: 1rem;
}

.feature span {
    color: var(--text-soft);
    font-size: 0.92rem;
    line-height: 1.5;
}

.panel-card {
    border-radius: 24px;
    border: 1px solid var(--panel-border);
    background: var(--panel-bg);
    box-shadow: var(--shadow);
    padding: 22px;
}

.panel-title {
    margin: 0 0 10px;
    color: var(--text-main);
    font-size: 1.2rem;
    font-weight: 800;
}

.panel-caption {
    margin: 0 0 16px;
    color: var(--text-soft);
    font-size: 0.95rem;
    line-height: 1.55;
}

.gradio-container .wrap {
    border: none !important;
    box-shadow: none !important;
}

.gradio-container .gr-button.primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 800 !important;
    box-shadow: 0 12px 24px rgba(255, 122, 24, 0.35);
}

.gradio-container .gr-button.secondary {
    background: rgba(148, 163, 184, 0.16) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    color: #e2e8f0 !important;
}

.gradio-container .wrap .input,
.gradio-container .wrap textarea,
.gradio-container .wrap input,
.gradio-container .wrap .select,
.gradio-container .wrap .dropdown,
.gradio-container .wrap .textbox {
    border-radius: 16px !important;
}

.results-note {
    margin-top: 14px;
    padding: 12px 14px;
    border-left: 4px solid var(--accent);
    border-radius: 14px;
    background: rgba(255, 122, 24, 0.08);
    color: var(--text-soft);
    font-size: 0.92rem;
}

@media (max-width: 900px) {
    .feature-grid {
        grid-template-columns: 1fr;
    }
}
"""

with gr.Blocks() as app:
    with gr.Column(elem_id="app-shell"):
        with gr.Column(elem_classes=["hero-card"]):
            gr.Markdown(
                """
<div class="eyebrow">AI Medical Symptom Checker</div>
<h1 class="hero-title">Fast disease prediction with a cleaner, more readable interface</h1>
<p class="hero-subtitle">Pick one or more symptoms, then get the top predicted conditions with percentages, descriptions, precautions, and a combined summary. The interface is designed to be easier to scan on both desktop and mobile.</p>
<div class="feature-grid">
  <div class="feature"><strong>Searchable symptoms</strong><span>Type to filter and select the symptom chips you need.</span></div>
  <div class="feature"><strong>Ranked results</strong><span>See the top three diseases with confidence percentages.</span></div>
  <div class="feature"><strong>Collective summary</strong><span>Read a short combined response for the overall likely condition.</span></div>
</div>
                """,
                elem_classes=["hero-copy"],
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                with gr.Column(elem_classes=["panel-card"]):
                    gr.Markdown("<h2 class='panel-title'>Select symptoms</h2><p class='panel-caption'>Use the searchable dropdown to choose one or more symptoms. You can also clear everything and start again with the button on the right.</p>")
                    symptom_input = gr.Dropdown(
                        choices=symptom_choices,
                        multiselect=True,
                        allow_custom_value=True,
                        value=[],
                        label="Symptoms",
                        info="Search and select multiple symptoms"
                    )

                    with gr.Row():
                        submit_btn = gr.Button("Predict", variant="primary", scale=2)
                        clear_btn = gr.ClearButton(symptom_input, variant="secondary", scale=1)

                    gr.HTML("<div class='results-note'>Tip: select 3-5 related symptoms for better accuracy. The app returns the top 3 disease matches with percentages, then a collective summary of the most likely overall condition.</div>")

            with gr.Column(scale=7):
                with gr.Column(elem_classes=["panel-card"]):
                    gr.Markdown("<h2 class='panel-title'>Prediction results</h2><p class='panel-caption'>The response will appear here once you click Predict.</p>")
                    output_box = gr.Textbox(label="Results", lines=22, interactive=False)

        submit_btn.click(fn=predict, inputs=[symptom_input], outputs=output_box)

if __name__ == "__main__":
    app.launch(css=APP_CSS, theme=gr.themes.Soft())
