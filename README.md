# Multimodal Music Emotion Recognition (MER) & Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-informational.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end multimodal machine learning system that predicts musical emotions (**Happy, Angry, Calm, Sad**) by fusing **Spotify acoustic features** with **NLP lyric semantics** via a **Late Fusion Meta-Classifier**, deployed through an interactive **Streamlit web application**.

---

## 📌 Project Overview

Traditional music emotion recognition typically relies solely on acoustic audio signals or purely on lyrics. However, music is inherently multimodal: a track with upbeat acoustic rhythm can contain melancholy lyrics.

This project implements a **Late Fusion Multimodal Architecture**:
1. **Acoustic Stream:** Machine learning voting ensembles trained on Spotify audio descriptors (energy, valence, tempo, danceability, acousticness) mapped to Russell's Circumplex Model of Affect (Valence-Arousal space via the DEAM dataset).
2. **NLP Lyrics Stream:** Fine-tuned **DistilBERT** transformer language model extracting semantic emotion and lyrical sentiment.
3. **Late Fusion Classifier:** A probabilistic meta-classifier (Logistic Regression) dynamically blending predictions from both modalities, achieving superior classification performance over unimodal baselines (~75–82% vs. 70.19% unimodal acoustic).
4. **Interactive Web Deployment:** A full-featured web dashboard styled with a custom **Neo Kinpaku** dark gold design system, featuring real-time emotion radar charts, playlist emotion profiling, and intelligent mood-based music recommendations.

---

## 🏗️ System Architecture

```text
               ┌───────────────────────────────┐
               │         Input Song            │
               └──────┬─────────────────┬──────┘
                      │                 │
         [Acoustic Features]       [Song Lyrics]
                      │                 │
                      ▼                 ▼
          ┌───────────────────────┐ ┌───────────────────────┐
          │ Acoustic ML Ensemble  │ │ Fine-tuned DistilBERT │
          │ (RF / LGBM / Voting)  │ │ (Transformer Model)   │
          └───────────┬───────────┘ └───────────┬───────────┘
                      │                         │
            P(Emotion | Audio)        P(Emotion | Lyrics)
                      │                         │
                      └────────────┬────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Late Fusion Meta-Classifier │
                    │     (Logistic Regression)    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                     Predicted Emotion Quadrant
                     (Happy / Calm / Angry / Sad)
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Streamlit Web Dashboard & UI │
                    │ (Plotly Radar, Song Recomm.) │
                    └──────────────────────────────┘
```

---

## 🗂️ Repository Structure

```text
├── 01_Data_Preparation_and_EDA.py    # Data cleaning, EDA, DEAM annotation mapping & ANOVA
├── 02_Acoustic_Machine_Learning.py   # Acoustic feature engineering, model tuning & voting ensemble
├── 03_NLP_Lyrics_Pipeline.py         # Text preprocessing, tokenization & DistilBERT fine-tuning
├── 04_Late_Fusion_Classifier.py      # Multimodal late fusion blending & meta-classifier training
├── 05_Cross_Validation.py            # Stratified K-Fold validation & generalizability evaluation
├── 06_Streamlit_Deployment.py        # Production Streamlit web application with Neo Kinpaku UI
├── static_annotations_*.csv          # DEAM valence-arousal benchmark annotations
├── test_playlist.csv                 # Evaluation test sample playlist
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes large CSV datasets & model checkpoint files
└── README.md                         # Project documentation
```

---

## 📊 Emotion Quadrant Mapping (Russell's Circumplex Model)

| Quadrant | Valence | Arousal | Musical Emotion | Example Characteristics |
|---|---|---|---|---|
| **Q1** | High (+) | High (+) | **Happy** / Energetic | Upbeat pop, high danceability, uplifting lyrics |
| **Q2** | Low (-) | High (+) | **Angry** / Intense | Heavy rock, aggressive rhythm, intense delivery |
| **Q3** | Low (-) | Low (-) | **Sad** / Melancholic | Acoustic ballad, minor keys, somber lyrics |
| **Q4** | High (+) | Low (-) | **Calm** / Relaxed | Ambient, chill acoustic, peaceful sentiment |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd <YOUR-REPO-NAME>
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run 06_Streamlit_Deployment.py
```

The interactive dashboard will launch at `http://localhost:8501`.

---

## 💻 Tech Stack

- **Machine Learning & NLP:** PyTorch, Hugging Face Transformers (`DistilBERT`), Scikit-Learn, SciPy
- **Data Engineering & Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Web App & Visualization:** Streamlit, Plotly Express/Graph Objects, Streamlit Option Menu, Streamlit Lottie

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
