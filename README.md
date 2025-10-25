# NLP-LegalBERT-JudgementPrediction

A web-based application to **predict legal case outcomes** using a fine-tuned **Legal-BERT model**.

---

## Project Overview

Legal case analysis is a complex and time-consuming task. Lawyers and legal professionals often have to read through extensive documents to assess potential outcomes. This project aims to **automate preliminary judgment prediction** using natural language processing (NLP) techniques, providing an **AI-assisted decision support tool**.

The system leverages **Legal-BERT**, a BERT model pre-trained on legal corpora, and fine-tunes it on labeled case data to predict whether a case is likely to **favor the plaintiff (violation)** or the **defendant (no violation)**.

---

## Motivation

- Reduce the **time and effort** needed to assess legal case outcomes.  
- Provide **data-driven insights** for legal professionals.  
- Explore applications of NLP in the **legal domain**, which is traditionally slow to adopt AI.  

---

## Dataset

### Current Dataset (Synthetic/Simulated)

- A **synthetic dataset** was created to simulate real-world legal cases.  
- Contains **5,000 cases**:
  - 4,000 training samples  
  - 500 validation samples  
  - 500 test samples  
- Balanced classes: 50% favor plaintiff, 50% favor defendant.  
- Each case includes text describing the legal situation and a label (`0` for Defendant, `1` for Plaintiff).

### Ideal Dataset

- Real-world **court case judgments** or **legal case summaries**.  
- Should be labeled according to actual court outcomes (plaintiff vs. defendant).  
- Could be sourced from public legal datasets, e.g., **CaseHOLD**, **European Court of Human Rights datasets**, or other open-access judgment datasets.  

---

## Approach

1. **Data Cleaning & Preprocessing**  
   - Remove corrupted or very short texts.  
   - Normalize whitespace and text formatting.  

2. **Model**  
   - **Legal-BERT (`nlpaueb/legal-bert-base-uncased`)** for sequence classification.  
   - Fine-tuned on the synthetic dataset with 2 output classes.  

3. **Training & Evaluation**  
   - 3 epochs with batch size 8, using GPU if available.  
   - Evaluated using **accuracy, precision, recall, F1-score**.  
   - Achieved near-perfect performance on synthetic dataset (expected, since data is clean and balanced).  

4. **Deployment**  
   - Streamlit-based frontend for users to input case text.  
   - Model predicts judgment outcome with a confidence score.  

---

## Features

- Simple web interface to **paste case text** and get predictions.  
- Displays **predicted outcome** and **confidence**.  
- Fully integrated **Local/Offline model** (no API calls required).  

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/noeljk03/NLP-LegalBERT-JudgementPrediction.git
cd NLP-LegalBERT-JudgementPrediction/LegalBertAPP
```

2. Create and activate a Python virtual environment:

```bash
python -m venv legalbert-env
# Windows
legalbert-env\Scripts\activate
# Mac/Linux
source legalbert-env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### How to Use

1. Open the URL provided in the terminal (usually `http://localhost:8501`).
2. Paste a **legal case description** in the text area.
3. Click **Predict** to see:
   - **Outcome:** Favor Plaintiff (Violation) / Favor Defendant (No Violation)
   - **Confidence Score**

---

### Limitations

- Current dataset is **synthetic**, so the model may not generalize well to real legal data.  
- Real-world deployment requires **legally sourced, labeled datasets**.  
- Predictions are **for research and guidance only** — not to be used as **legal advice**.  

---

### Future Improvements

- Train on **real-world court judgment datasets** for better generalization.  
- Extend model to **multi-class predictions** (e.g., civil, criminal, contract disputes).  
- Integrate **explainable AI (XAI)** features to highlight influential text segments.  
- Deploy as a **secure cloud-based web service** with a modern UI and API support.
