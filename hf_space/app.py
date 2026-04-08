import os
import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model
from preprocess import clean_text 

# Memuat model
model_path = "nlp_pipeline_final"
model = load_model(model_path)

def predict_sentiment(review):
    cleaned_review = clean_text(review)
    if not cleaned_review:
        return "Teks tidak valid."
    df_input = pd.DataFrame({'cleaned_text': [cleaned_review]})
    predictions = predict_model(model, data=df_input)
    
    if 'prediction_label' in predictions.columns:
        sentiment = predictions['prediction_label'].iloc[0]
    else:
        sentiment = predictions['Label'].iloc[0]
    return f"Sentimen: {sentiment.upper()}"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Contoh: Barangnya bagus banget!"),
    outputs="text",
    title="Analisis Sentimen E-commerce Indonesia",
    description="Masukkan ulasan produk untuk melihat hasil prediksi sentimen (Positif/Negatif)."
)

if __name__ == "__main__":
    demo.launch()