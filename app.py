import os
import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# Import fungsi pembersih dari file punyamu
from preprocess import clean_text 

# Memuat model yang baru saja kamu buat tadi
# PyCaret otomatis akan mencari file di folder 'models/nlp_pipeline_final.pkl'
model_path = os.path.join("models", "nlp_pipeline_final")
model = load_model(model_path)

def predict_sentiment(review):
    # 1. Bersihkan teks input
    cleaned_review = clean_text(review)
    
    if not cleaned_review:
        return "Teks tidak valid."

    # 2. Buat DataFrame untuk input PyCaret
    df_input = pd.DataFrame({'cleaned_text': [cleaned_review]})
    
    # 3. Prediksi
    predictions = predict_model(model, data=df_input)
    
    # 4. Ambil hasil (biasanya kolom 'prediction_label')
    if 'prediction_label' in predictions.columns:
        sentiment = predictions['prediction_label'].iloc[0]
    else:
        sentiment = predictions['Label'].iloc[0]
        
    return f"Sentimen: {sentiment.upper()}"

# Membuat Tampilan Website Sederhana
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Contoh: Barangnya bagus banget, pengiriman cepat!"),
    outputs="text",
    title="Indonesian E-commerce Sentiment Analysis",
    description="Tugas Besar Machine Learning - Masukkan ulasan untuk cek sentimennya."
)

if __name__ == "__main__":
    demo.launch()