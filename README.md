# 🛡️ URL Phishing Classifier using NLP + Streamlit

A Machine Learning-based web application for detecting phishing URLs using Natural Language Processing techniques. Built with Python, TensorFlow, Scikit-learn, and deployed using Streamlit.

## 📁 Project Structure

```
url-phishing-classifier/
├── data/                   # Raw and cleaned datasets
├── notebooks/              # Jupyter notebooks for EDA and model experimentation
├── src/                    # Core source code
│   ├── preprocessing.py    # Feature extraction and cleaning
│   ├── model.py            # Model architecture and training functions
│   ├── evaluate.py         # Evaluation metrics and plots
├── app/                    # Streamlit application
│   └── streamlit_app.py    # Streamlit UI logic
├── models/                 # Saved models (H5 / PKL)
├── requirements.txt        # Dependencies
├── README.md               # Project overview
└── utils.py                # Helper functions
```

## 🔧 Tech Stack & Tools

| Component     | Tool/Library                            |
| ------------- | --------------------------------------- |
| Language      | Python                                  |
| NLP           | TensorFlow, Scikit-learn                |
| Feature Eng.  | URL tokenization (Tokinizer), TF-IDF    |
| DL Models     | LSTM-CNN, LSTM, CNN, BiLSTM             |
| Frontend UI   | Streamlit                               |
| EDA/Notebooks | Pandas, Seaborn, Matplotlib             |
| Deployment    | Streamlit Cloud / Local Docker          |

## 🚀 Features

* Detects phishing URLs based on lexical and token patterns
* Lightweight Streamlit web interface for easy testing
* Visualizes evaluation metrics (Training Curve, Confusion Matrix, Clasification Report, .)
* Easy to retrain on new data

## 📊 Workflow

1. 📥 Data Collection
   → Dataset of phishing and legitimate URLs (e.g., Kaggle, PhishTank)

2. 🧹 Preprocessing
   → Clean URLs, extract tokens, encode features (TF-IDF / Tokenizer)

3. 🧠 Model Training
   → Train various classifiers (LSTM-CNN, LSTM, CNN, BiLSTM)

4. 📈 Evaluation
   → Evaluate with Accuracy, F1, ROC-AUC, Confusion Matrix

5. 🖥️ Streamlit Deployment
   → Build interactive UI to predict and visualize results

## 📦 Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/url-phishing-classifier.git
cd url-phishing-classifier
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## ✅ To-Do

* [ ] Integrate live URL fetch and scan
* [ ] Containerize with Docker
* [ ] Add language support for multiple locales

## 📄 License

MIT License. See LICENSE file for details.

---

Let me know if you'd like me to generate a requirements.txt or .streamlit/config.toml next?
# NLP-Project-URL-Phishing-Classifier
