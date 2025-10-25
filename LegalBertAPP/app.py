import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch.nn.functional as F

# --------------------------
# Load Model
# --------------------------
MODEL_PATH = "./final_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource  # Streamlit caches the model so it doesn't reload every run
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model(MODEL_PATH)

# --------------------------
# Prediction Function
# --------------------------
def predict(text: str):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Legal-BERT Judgment Predictor", layout="wide")
st.title("⚖️ Legal-BERT Judgment Predictor")
st.write("Paste a legal case and get the predicted judgment outcome.")

user_input = st.text_area("Enter case text here:", height=250)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Predicting..."):
            pred_class, confidence = predict(user_input)
            outcome = "Favor Plaintiff (Violation)" if pred_class == 1 else "Favor Defendant (No Violation)"
            st.success(f"Prediction: {outcome}")
            st.info(f"Confidence: {confidence:.2%}")
