# Sub-module: validation_1_deam_crossval.py

"""
=============================================================================
VALIDATION SCRIPT 1: DEAM Cross-Validation
=============================================================================
Purpose:
  Apply our quadrant mapping to DEAM's HUMAN-annotated valence/arousal scores
  and compare the resulting label distribution + inter-annotator agreement
  against our machine-derived Spotify labels.

This script does NOT require matching songs by name. Instead it:
  1. Maps DEAM human VA scores -> emotion quadrants (same logic we used)
  2. Analyses how much humans AGREED with each other (via std scores)
  3. Compares DEAM label distribution vs our Spotify label distribution
  4. Calculates what % of DEAM songs are "ambiguous" (near the VA centre)
  5. Outputs a clean summary for your FYP report
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load and combine DEAM files
# ─────────────────────────────────────────────────────────────────────────────
print("Loading DEAM dataset...")
d1 = pd.read_csv(os.path.join(BASE_DIR, "DEAM Dataset", "static_annotations_averaged_songs_1_2000.csv"))
d2 = pd.read_csv(os.path.join(BASE_DIR, "DEAM Dataset", "static_annotations_averaged_songs_2000_2058.csv"))
deam = pd.concat([d1, d2], ignore_index=True)

# Clean up column name spaces
deam.columns = deam.columns.str.strip()
print(f"  ✓ DEAM loaded: {len(deam)} songs")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Apply quadrant mapping to DEAM (scale is 1-9, midpoint = 5)
# ─────────────────────────────────────────────────────────────────────────────
DEAM_MID = 5.0  # Midpoint of the 1-9 annotation scale

def deam_quadrant(valence, arousal):
    if valence >= DEAM_MID and arousal >= DEAM_MID:
        return "Happy"
    elif valence < DEAM_MID and arousal >= DEAM_MID:
        return "Angry"
    elif valence >= DEAM_MID and arousal < DEAM_MID:
        return "Calm"
    else:
        return "Sad"

deam["emotion_label"] = deam.apply(
    lambda r: deam_quadrant(r["valence_mean"], r["arousal_mean"]), axis=1
)

print("\n[STEP 2] DEAM Emotion Label Distribution (Human-Annotated):")
deam_dist = deam["emotion_label"].value_counts()
deam_pct  = (deam_dist / len(deam) * 100).round(1)
for emotion in ["Happy", "Angry", "Calm", "Sad"]:
    count = deam_dist.get(emotion, 0)
    pct   = deam_pct.get(emotion, 0)
    print(f"  {emotion:<8}: {count:>4} songs ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Inter-annotator agreement analysis (std per class)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Inter-Annotator Agreement per Emotion (lower std = more agreement):")
agreement = deam.groupby("emotion_label")[["valence_std", "arousal_std"]].mean().round(3)
print(agreement.to_string())

overall_v_std = deam["valence_std"].mean()
overall_a_std = deam["arousal_std"].mean()
print(f"\n  Overall average Valence std: {overall_v_std:.3f}")
print(f"  Overall average Arousal std: {overall_a_std:.3f}")
print(f"  (On a 1-9 scale — lower = humans agreed more)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Ambiguity analysis — how many songs are near the VA centre?
# ─────────────────────────────────────────────────────────────────────────────
AMBIGUITY_THRESHOLD = 0.75  # Within 0.75 units of midpoint = ambiguous
deam["v_dist_from_mid"] = abs(deam["valence_mean"] - DEAM_MID)
deam["a_dist_from_mid"] = abs(deam["arousal_mean"] - DEAM_MID)
deam["is_ambiguous"] = (
    (deam["v_dist_from_mid"] < AMBIGUITY_THRESHOLD) |
    (deam["a_dist_from_mid"] < AMBIGUITY_THRESHOLD)
)

n_ambiguous = deam["is_ambiguous"].sum()
pct_ambiguous = n_ambiguous / len(deam) * 100
print(f"\n[STEP 4] Ambiguity Analysis:")
print(f"  Songs near VA boundary (within {AMBIGUITY_THRESHOLD} units of midpoint): "
      f"{n_ambiguous} / {len(deam)} ({pct_ambiguous:.1f}%)")
print(f"  → These are songs where even humans found it hard to agree on the emotion.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Load Spotify dataset and compare label distributions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Loading Spotify dataset for distribution comparison...")
spotify = pd.read_csv(os.path.join(BASE_DIR, "spotify_prepared_final.csv"),
                      usecols=["emotion_label"])
spotify_dist = spotify["emotion_label"].value_counts()
spotify_pct  = (spotify_dist / len(spotify) * 100).round(1)

print("\n  Spotify (Machine-Derived) Label Distribution:")
for emotion in ["Happy", "Angry", "Calm", "Sad"]:
    count = spotify_dist.get(emotion, 0)
    pct   = spotify_pct.get(emotion, 0)
    print(f"  {emotion:<8}: {count:>6} songs ({pct:.1f}%)")

print(f"\n  DEAM (Human-Annotated) Label Distribution:")
for emotion in ["Happy", "Angry", "Calm", "Sad"]:
    count = deam_dist.get(emotion, 0)
    pct   = deam_pct.get(emotion, 0)
    print(f"  {emotion:<8}: {count:>4} songs ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Plot comparison chart
# ─────────────────────────────────────────────────────────────────────────────
emotions = ["Happy", "Angry", "Calm", "Sad"]
x = np.arange(len(emotions))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2,
               [spotify_pct.get(e, 0) for e in emotions],
               width, label="Spotify (Machine-Derived)", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width/2,
               [deam_pct.get(e, 0) for e in emotions],
               width, label="DEAM (Human-Annotated)", color="#DD8452", alpha=0.85)

ax.set_xlabel("Emotion Category", fontsize=13)
ax.set_ylabel("Percentage of Songs (%)", fontsize=13)
ax.set_title("Label Distribution: Machine-Derived (Spotify) vs Human-Annotated (DEAM)",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(emotions, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 55)
for bar in bars1:
    ax.annotate(f"{bar.get_height():.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "deam_vs_spotify_distribution.png")
plt.savefig(out_path, dpi=150)
print(f"\n  ✓ Chart saved to: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Plot VA scatter for DEAM
# ─────────────────────────────────────────────────────────────────────────────
colour_map = {"Happy": "#2ecc71", "Angry": "#e74c3c", "Calm": "#3498db", "Sad": "#9b59b6"}
fig2, ax2 = plt.subplots(figsize=(9, 7))
for emotion in emotions:
    subset = deam[deam["emotion_label"] == emotion]
    ax2.scatter(subset["valence_mean"], subset["arousal_mean"],
                c=colour_map[emotion], label=emotion, alpha=0.6, s=40, edgecolors="none")

ax2.axvline(x=DEAM_MID, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
ax2.axhline(y=DEAM_MID, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
ax2.set_xlabel("Valence (1=Negative  →  9=Positive)", fontsize=12)
ax2.set_ylabel("Arousal (1=Calm  →  9=Excited)", fontsize=12)
ax2.set_title("DEAM Human-Annotated Songs in Valence-Arousal Space", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.set_xlim(1, 9); ax2.set_ylim(1, 9)

# Quadrant labels
ax2.text(7.5, 8.5, "Happy", fontsize=11, color="#2ecc71", fontweight="bold")
ax2.text(1.5, 8.5, "Angry", fontsize=11, color="#e74c3c", fontweight="bold")
ax2.text(7.5, 1.5, "Calm",  fontsize=11, color="#3498db", fontweight="bold")
ax2.text(1.5, 1.5, "Sad",   fontsize=11, color="#9b59b6", fontweight="bold")

plt.tight_layout()
out_path2 = os.path.join(OUTPUT_DIR, "deam_va_scatter.png")
plt.savefig(out_path2, dpi=150)
print(f"  ✓ VA scatter saved to: {out_path2}")

print("\n" + "="*60)
print("  DEAM VALIDATION COMPLETE")
print("="*60)

# Sub-module: validation_2_vader_crossval.py

"""
=============================================================================
VALIDATION SCRIPT 2: Lyrics Sentiment Cross-Validation (VADER)
=============================================================================
Purpose:
  Use VADER (a rule-based, linguistics-aware sentiment analyser) to
  independently score each song's lyrics and compare against our
  acoustic-derived emotion labels.

  Songs where VADER and acoustic labels AGREE = high-confidence samples
  Songs where VADER and acoustic labels DISAGREE = potentially noisy labels

  This script:
  1. Installs VADER if needed
  2. Scores all 16,752 lyrics
  3. Maps VADER scores to emotion quadrants
  4. Compares with acoustic labels and reports agreement rate
  5. Exports a "clean" high-confidence dataset with noisy samples flagged
=============================================================================
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "vaderSentiment"], stdout=subprocess.DEVNULL)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR  = r"C:\Users\User\OneDrive - Asia Pacific University\FYP"
LYRICS_CSV = os.path.join(BASE_DIR, "spotify_lyrics_balanced_final.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load lyrics dataset
# ─────────────────────────────────────────────────────────────────────────────
print("Loading lyrics dataset...")
df = pd.read_csv(LYRICS_CSV).dropna(subset=["lyrics"])
print(f"  ✓ Loaded {len(df):,} songs")
print(f"  Acoustic label distribution:\n{df['emotion_label'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: VADER sentiment scoring
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Running VADER sentiment analysis on all lyrics...")
analyzer = SentimentIntensityAnalyzer()

results = []
total = len(df)
for i, (_, row) in enumerate(df.iterrows()):
    scores = analyzer.polarity_scores(str(row["lyrics"])[:5000])  # cap at 5k chars
    results.append({
        "compound": scores["compound"],  # -1 (negative) to +1 (positive)
        "pos":      scores["pos"],
        "neg":      scores["neg"],
        "neu":      scores["neu"],
    })
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1:,}/{total:,}...")

vader_df = pd.DataFrame(results)
df = pd.concat([df.reset_index(drop=True), vader_df], axis=1)
print(f"  ✓ VADER scoring complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Map VADER compound score to emotion category
# VADER gives a single compound score (-1 to +1) = Valence proxy
# We use the neu score as an Arousal proxy (low neutrality = more emotional intensity)
# ─────────────────────────────────────────────────────────────────────────────
def vader_to_emotion(compound, neu):
    """
    compound > 0  → positive valence (Happy/Calm)
    compound <= 0 → negative valence (Angry/Sad)
    neu < 0.5     → high arousal/intensity (Happy/Angry)
    neu >= 0.5    → low arousal/neutral (Calm/Sad)
    """
    if compound > 0 and neu < 0.5:
        return "Happy"
    elif compound <= 0 and neu < 0.5:
        return "Angry"
    elif compound > 0 and neu >= 0.5:
        return "Calm"
    else:
        return "Sad"

df["vader_emotion"] = df.apply(
    lambda r: vader_to_emotion(r["compound"], r["neu"]), axis=1
)

print("\n[STEP 3] VADER-Derived Label Distribution:")
vader_dist = df["vader_emotion"].value_counts()
for e in ["Happy", "Angry", "Calm", "Sad"]:
    print(f"  {e:<8}: {vader_dist.get(e, 0):>5} ({vader_dist.get(e, 0)/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Agreement analysis
# ─────────────────────────────────────────────────────────────────────────────
df["agree"] = df["emotion_label"] == df["vader_emotion"]
overall_agreement = df["agree"].mean() * 100

print(f"\n[STEP 4] Overall Agreement Rate (Acoustic vs VADER): {overall_agreement:.1f}%")

print("\n  Per-class agreement:")
for emotion in ["Happy", "Angry", "Calm", "Sad"]:
    subset = df[df["emotion_label"] == emotion]
    rate   = subset["agree"].mean() * 100
    print(f"  {emotion:<8}: {rate:.1f}% ({subset['agree'].sum()} / {len(subset)})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Confidence scoring — flag high vs low confidence samples
# ─────────────────────────────────────────────────────────────────────────────
# A high-confidence sample = VADER compound score is far from zero
df["vader_confidence"] = abs(df["compound"])

HIGH_CONF_THRESHOLD = 0.3   # |compound| > 0.3 = reasonably confident
df["high_confidence"] = df["vader_confidence"] >= HIGH_CONF_THRESHOLD

n_high_conf  = df["high_confidence"].sum()
n_agree_hc   = df[df["high_confidence"]]["agree"].sum()
agree_hc_pct = n_agree_hc / n_high_conf * 100

print(f"\n[STEP 5] High-Confidence Samples (|compound| >= {HIGH_CONF_THRESHOLD}):")
print(f"  Total high-confidence: {n_high_conf:,} / {len(df):,} ({n_high_conf/len(df)*100:.1f}%)")
print(f"  Agreement in high-confidence subset: {agree_hc_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Export clean dataset (songs where VADER agrees with acoustic label)
# ─────────────────────────────────────────────────────────────────────────────
clean_df = df[df["agree"]].copy()
noisy_df = df[~df["agree"]].copy()

clean_path = os.path.join(BASE_DIR, "spotify_lyrics_clean_validated.csv")
noisy_path = os.path.join(BASE_DIR, "spotify_lyrics_noisy_flagged.csv")

# Save only the original columns + vader info
save_cols = [c for c in df.columns if c in
             ["artist", "Song-Name", "emotion_label", "lyrics",
              "compound", "vader_emotion", "vader_confidence", "agree"]]
clean_df[save_cols].to_csv(clean_path, index=False)
noisy_df[save_cols].to_csv(noisy_path, index=False)

print(f"\n[STEP 6] Exported files:")
print(f"  ✓ Clean (agreed) samples:  {len(clean_df):,} songs → {clean_path}")
print(f"  ✓ Noisy (disagreed) samples: {len(noisy_df):,} songs → {noisy_path}")
print(f"\n  Class distribution in CLEAN dataset:")
print(clean_df["emotion_label"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Agreement bar chart
# ─────────────────────────────────────────────────────────────────────────────
emotions = ["Happy", "Angry", "Calm", "Sad"]
agree_rates = []
for emotion in emotions:
    subset = df[df["emotion_label"] == emotion]
    agree_rates.append(subset["agree"].mean() * 100)

colours = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(emotions, agree_rates, color=colours, alpha=0.85, edgecolor="white", linewidth=1.2)
ax.axhline(y=overall_agreement, color="black", linestyle="--",
           linewidth=1.5, label=f"Overall avg: {overall_agreement:.1f}%")
ax.set_xlabel("Emotion Category", fontsize=13)
ax.set_ylabel("VADER Agreement Rate (%)", fontsize=13)
ax.set_title("Agreement Between Acoustic-Derived Labels and VADER Sentiment\n(Per Emotion Class)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 80)
ax.legend(fontsize=11)
for bar, rate in zip(bars, agree_rates):
    ax.annotate(f"{rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "vader_agreement_chart.png")
plt.savefig(chart_path, dpi=150)
print(f"\n  ✓ Agreement chart saved to: {chart_path}")

print("\n" + "="*60)
print("  VADER LYRICS VALIDATION COMPLETE")
print("="*60)
