# Sub-module: stage_5c_late_fusion.py

"""
=============================================================================
STAGE 5C: LATE FUSION — MULTIMODAL EMOTION CLASSIFIER
=============================================================================
This script combines two separate models into one powerful predictor:
  1. Acoustic Model  (Stage 4F):  Trained on musical features (tempo,
                                  energy, valence, genre, etc.)
  2. NLP Lyrics Model (Stage 5B): DistilBERT fine-tuned on song lyrics

Strategy: "Late Fusion" — both models make their own probability predictions
           independently, then a meta-classifier (Logistic Regression) learns
           the optimal way to BLEND those predictions for the final answer.

Expected Result: ~75-82% accuracy (vs 70.19% from the acoustic-only model)
=============================================================================
"""

import sys
import io
import os
import warnings
import numpy as np
import pandas as pd
import pickle
import joblib
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = r"C:\Users\User\OneDrive - Asia Pacific University\FYP"
LYRICS_CSV    = os.path.join(BASE_DIR, "spotify_lyrics_balanced_final.csv")
DISTILBERT_PATH = os.path.join(BASE_DIR, "final_distilbert_lyrics_2", "final_distilbert_lyrics")
ACOUSTIC_MODEL_PATH = os.path.join(BASE_DIR, "stage4f_voting_ensemble.pkl")
FUSION_OUTPUT = os.path.join(BASE_DIR, "stage5c_fusion_model.pkl")

EMOTION_MAP   = {'Angry': 0, 'Calm': 1, 'Happy': 2, 'Sad': 3}
INV_EMOTION   = {v: k for k, v in EMOTION_MAP.items()}
LABEL_ORDER   = ['Angry', 'Calm', 'Happy', 'Sad']

print("=" * 70)
print("STAGE 5C: LATE FUSION MULTIMODAL CLASSIFIER")
print("=" * 70)
print(f"\nDevice: {'CUDA GPU' if torch.cuda.is_available() else 'CPU (slower, still works)'}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load & Prepare the Lyrics Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 1] Loading balanced lyrics dataset...")
df = pd.read_csv(LYRICS_CSV)
df = df.dropna(subset=['lyrics', 'emotion_label'])
df = df[df['emotion_label'].isin(EMOTION_MAP.keys())]
df['label'] = df['emotion_label'].map(EMOTION_MAP)

# Use same 80/20 split as training (same random_state ensures no data leakage)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
print(f"  ✓ Train size: {len(train_df)}, Test size: {len(test_df)}")
print(f"  ✓ Test class distribution:\n{test_df['emotion_label'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load DistilBERT and Get Lyric Probability Predictions
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 2] Loading fine-tuned DistilBERT from:\n  {DISTILBERT_PATH}")
tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
nlp_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp_model = nlp_model.to(device)
nlp_model.eval()

def get_nlp_probabilities(texts, batch_size=32):
    """Run DistilBERT inference and return probability arrays."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i+batch_size])
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = nlp_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if (i // batch_size) % 5 == 0:
            print(f"    Processed {min(i+batch_size, len(texts))}/{len(texts)} lyrics...", flush=True)
    return np.vstack(all_probs)

print("\n  Running DistilBERT on TRAIN set...")
train_nlp_probs = get_nlp_probabilities(train_df['lyrics'].values)
print("  Running DistilBERT on TEST set...")
test_nlp_probs  = get_nlp_probabilities(test_df['lyrics'].values)
print("  ✓ NLP probability extraction complete!")

# Quick sanity check: NLP-only accuracy
train_nlp_preds = np.argmax(train_nlp_probs, axis=1)
test_nlp_preds  = np.argmax(test_nlp_probs,  axis=1)
nlp_test_acc = accuracy_score(test_df['label'].values, test_nlp_preds)
print(f"\n  📊 NLP-only accuracy on test set: {nlp_test_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build Acoustic Features for the same songs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] Building acoustic feature vectors...")

# NOTE: 'valence' is intentionally EXCLUDED from acoustic features.
# The emotion labels (Angry/Calm/Happy/Sad) were derived from the
# Valence-Arousal model using valence as input. Including valence here
# would create circular data leakage (perfect 100% accuracy but meaningless).
ACOUSTIC_FEATURES = [
    'Energy', 'Danceability', 'Tempo', 'Loudness',
    'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
    'key_num', 'mode', 'Popularity'
]
GENRE_COL = 'Genre'

# Map broad genre groups (same as Stage 4F to match the model's expectations)
GENRE_MAP = {
    'pop': 'pop', 'indie pop': 'pop', 'electropop': 'pop', 'synth-pop': 'pop',
    'k-pop': 'pop', 'j-pop': 'pop', 'dance pop': 'pop',
    'hip hop': 'hip_hop', 'rap': 'hip_hop', 'trap': 'hip_hop',
    'r&b': 'hip_hop', 'southern hip hop': 'hip_hop', 'alternative hip hop': 'hip_hop',
    'rock': 'rock', 'alternative rock': 'rock', 'indie rock': 'rock',
    'hard rock': 'rock', 'punk rock': 'rock', 'post-punk': 'rock',
    'metal': 'metal', 'heavy metal': 'metal', 'death metal': 'metal',
    'metalcore': 'metal', 'black metal': 'metal',
    'electronic': 'electronic', 'edm': 'electronic', 'house': 'electronic',
    'techno': 'electronic', 'dubstep': 'electronic', 'ambient': 'electronic',
    'country': 'country', 'country road': 'country', 'bluegrass': 'country',
    'folk': 'folk', 'indie folk': 'folk', 'singer-songwriter': 'folk',
    'jazz': 'jazz', 'soul': 'jazz', 'blues': 'jazz', 'funk': 'jazz',
    'classical': 'classical', 'opera': 'classical', 'orchestral': 'classical',
    'latin': 'latin', 'reggaeton': 'latin', 'salsa': 'latin',
    'reggae': 'reggae', 'dancehall': 'reggae',
}

def map_genre(genre_str):
    if pd.isna(genre_str):
        return 'other'
    genre_lower = str(genre_str).lower().strip()
    for key, val in GENRE_MAP.items():
        if key in genre_lower:
            return val
    return 'other'

def build_acoustic_features(data_df):
    feats = data_df[ACOUSTIC_FEATURES].copy().fillna(0)
    feats['genre_broad'] = data_df[GENRE_COL].apply(map_genre) if GENRE_COL in data_df.columns else 'other'
    genre_dummies = pd.get_dummies(feats['genre_broad'], prefix='genre')
    feats = pd.concat([feats.drop('genre_broad', axis=1), genre_dummies], axis=1)
    return feats

train_acoustic = build_acoustic_features(train_df)
test_acoustic  = build_acoustic_features(test_df)

# Align columns (test might have different genre dummies)
train_cols = set(train_acoustic.columns)
test_cols  = set(test_acoustic.columns)
for col in train_cols - test_cols:
    test_acoustic[col] = 0
test_acoustic = test_acoustic[train_acoustic.columns]

print(f"  ✓ Acoustic feature matrix: {train_acoustic.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Train a Simple Acoustic-Only Baseline for Comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] Training acoustic-only baseline on this dataset...")
from sklearn.ensemble import RandomForestClassifier
acoustic_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
acoustic_clf.fit(train_acoustic, train_df['label'].values)
train_acoustic_probs = acoustic_clf.predict_proba(train_acoustic)
test_acoustic_probs  = acoustic_clf.predict_proba(test_acoustic)
acoustic_preds = np.argmax(test_acoustic_probs, axis=1)
acoustic_acc = accuracy_score(test_df['label'].values, acoustic_preds)
print(f"  📊 Acoustic-only accuracy on THIS dataset: {acoustic_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Late Fusion — Combine Both Sets of Probabilities
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 5] Training Late Fusion Meta-Classifier...")
print("  Strategy: Logistic Regression on [NLP_probs | Acoustic_probs] (8 features)")

# Stack both probability vectors side-by-side as input to the meta-classifier
X_train_fusion = np.hstack([train_nlp_probs, train_acoustic_probs])  # shape: (N, 8)
X_test_fusion  = np.hstack([test_nlp_probs,  test_acoustic_probs])   # shape: (N, 8)
y_train        = train_df['label'].values
y_test         = test_df['label'].values

fusion_clf = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    solver='lbfgs'
)
fusion_clf.fit(X_train_fusion, y_train)
fusion_preds = fusion_clf.predict(X_test_fusion)
fusion_acc   = accuracy_score(y_test, fusion_preds)

print(f"\n{'='*70}")
print(f"  RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  📊 NLP (Lyrics) Only:               {nlp_test_acc*100:.2f}%")
print(f"  📊 Acoustic Only (this dataset):     {acoustic_acc*100:.2f}%")
print(f"  🏆 Late Fusion (NLP + Acoustic):     {fusion_acc*100:.2f}%")
print(f"{'='*70}")

print("\n  Detailed Classification Report (Fusion Model):")
print(classification_report(
    y_test, fusion_preds,
    target_names=LABEL_ORDER,
    digits=3
))

print("\n  Confusion Matrix (rows=True, cols=Predicted):")
cm = confusion_matrix(y_test, fusion_preds)
cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
print(cm_df.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Save the Fusion Model Components
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 6] Saving fusion model to: {FUSION_OUTPUT}")
fusion_package = {
    'meta_classifier': fusion_clf,
    'acoustic_classifier': acoustic_clf,
    'acoustic_train_columns': list(train_acoustic.columns),
    'emotion_map': EMOTION_MAP,
    'inv_emotion': INV_EMOTION,
    'label_order': LABEL_ORDER,
    'acoustic_features': ACOUSTIC_FEATURES,
    'distilbert_path': DISTILBERT_PATH,
    'fusion_test_accuracy': fusion_acc,
    'nlp_only_accuracy': nlp_test_acc,
    'acoustic_only_accuracy': acoustic_acc,
}
joblib.dump(fusion_package, FUSION_OUTPUT)
print(f"  ✓ Saved successfully!")

print(f"\n{'='*70}")
print(f"  STAGE 5C COMPLETE!")
print(f"  Final Fusion Accuracy: {fusion_acc*100:.2f}%")
print(f"{'='*70}\n")
