import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# ---------------------------------------------------------------------
# Helper functions assumed to exist in the original notebook environment.
# Included here so this file runs standalone; remove if you already have
# your own versions defined elsewhere in your project.
# ---------------------------------------------------------------------

FIG_DIR = "figures"

def save_fig(n, label):
    """Save the current matplotlib figure to FIG_DIR as '{n:03d}_{label}.png'."""
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    safe_label = label.replace(" ", "_")
    plt.savefig(f"{FIG_DIR}/{n:03d}_{safe_label}.png", bbox_inches="tight", dpi=150)
    plt.show()

def render_table(df, figsize=(10, 6), fontsize=10):
    """Render a DataFrame as a formatted table image."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.5)

emotion_palette = {"Happy": "#E8C547", "Angry": "#C43D53", "Calm": "#4C7EA8", "Sad": "#7C8798"}

# 4.4.1  VARIABLES  -  loading and structural inspection

# Loading the Spotify dataset and extracting structural information
spotify_df = pd.read_csv("light_spotify_dataset.csv")
spotify_df.columns = spotify_df.columns.str.strip()
spotify_df = spotify_df.rename(columns={"song": "Song-Name"})
spotify_df.info()

# Previewing the first five records of the Spotify dataset
render_table(spotify_df.head(), figsize=(18, 3), fontsize=8)
save_fig(2, "Spotify dataset head")

# Compiling the variable description table for the Spotify dataset
variable_table = pd.DataFrame({
    "Column": spotify_df.columns,
    "Type": spotify_df.dtypes.astype(str).values,
})
render_table(variable_table, figsize=(8, 12), fontsize=10)
save_fig(7, "Spotify variable description table")

# Loading DEAM File 1 and extracting structural information
deam_1 = pd.read_csv("static_annotations_averaged_songs_1_2000.csv")
deam_1.columns = deam_1.columns.str.strip()
deam_1.info()

# Previewing the first five records of DEAM File 1
render_table(deam_1.head(), figsize=(10, 3), fontsize=9)
save_fig(10, "DEAM File 1 head")

# Loading DEAM File 2 and extracting structural information
deam_2 = pd.read_csv("static_annotations_averaged_songs_2000_2058.csv")
deam_2.columns = deam_2.columns.str.strip()
deam_2.info()

# Merging DEAM File 1 and File 2 and verifying the combined shape
deam_df = pd.concat([deam_1, deam_2], ignore_index=True)
print(f"Combined DEAM shape: {deam_df.shape}")

# Previewing the first five records of the merged DEAM dataframe
render_table(deam_df.head(), figsize=(16, 3), fontsize=8)
save_fig(14, "merged DEAM head")

# 4.4.2  OBSERVATIONS  -  dataset dimensions

# Verifying the raw Spotify dataset shape
print(f"Spotify dataset shape: {spotify_df.shape}")

# 4.4.3  DESCRIPTIVE STATISTICS

# Generating descriptive statistics for the Spotify dataset
spotify_desc = spotify_df.describe()
render_table(spotify_desc.reset_index(), figsize=(20, 4), fontsize=8)
save_fig(6, "Spotify describe table")

# Generating descriptive statistics for the combined DEAM dataset
deam_desc = deam_df.describe()
render_table(deam_desc.reset_index(), figsize=(20, 4), fontsize=8)
save_fig(15, "DEAM describe table")

# 4.3.1  MISSING VALUES

# Null value check across all 18 columns
null_counts = spotify_df.isnull().sum().reset_index()
null_counts.columns = ["Column", "Null Count"]
render_table(null_counts, figsize=(10, 8), fontsize=11)
save_fig(16, "null value check")

# Isolating rows where Song-Name is null
missing_song_name = spotify_df[spotify_df["Song-Name"].isnull()]
render_table(missing_song_name, figsize=(14, 4), fontsize=9)
save_fig(17, "missing Song-Name rows")

# Dropping the 8 null Song-Name rows and verifying shape
spotify_df = spotify_df.dropna(subset=["Song-Name"])
print(f"Shape after dropping nulls: {spotify_df.shape}")

# Checking the count of fully duplicated rows
duplicate_count = spotify_df.duplicated().sum()
print(f"Duplicate row count: {duplicate_count}")

# Displaying the duplicate rows
duplicate_rows = spotify_df[spotify_df.duplicated(keep=False)]
render_table(duplicate_rows, figsize=(16, 4), fontsize=8)
save_fig(20, "duplicate rows identified")

# Dropping duplicate rows and verifying shape
spotify_df = spotify_df.drop_duplicates(keep="first")
print(f"Shape after dropping duplicates: {spotify_df.shape}")

# Post-clean null verification
post_clean_nulls = spotify_df.isnull().sum().reset_index()
post_clean_nulls.columns = ["Column", "Null Count"]
render_table(post_clean_nulls, figsize=(10, 8), fontsize=11)
save_fig(22, "post-clean null verification")

# 4.3.2  DIRTY DATA / CATEGORICAL ANOMALIES

# Checking unique values in the emotion column
emotion_counts = spotify_df["emotion"].value_counts().reset_index()
emotion_counts.columns = ["Label", "Count"]
render_table(emotion_counts, figsize=(8, 6), fontsize=11)
save_fig(24, "unique emotion values")

# Displaying rows carrying invalid emotion values
valid_labels = ["joy", "sadness", "anger", "love", "fear", "surprise"]
dirty_rows = spotify_df[~spotify_df["emotion"].isin(valid_labels)]
render_table(dirty_rows, figsize=(16, 5), fontsize=8)
save_fig(25, "dirty rows identified")

# Dropping the 10 dirty rows and verifying the final cleaned shape
spotify_df = spotify_df[spotify_df["emotion"].isin(valid_labels)]
print(f"Final cleaned shape: {spotify_df.shape}")

# 4.3.3.1  DATA TRANSFORMATION - KEY COLUMN DECOMPOSITION

# Extracting the unique Key values
unique_keys = spotify_df["Key"].unique()
render_table(pd.DataFrame(sorted(unique_keys), columns=["Key"]), figsize=(6, 10), fontsize=10)
save_fig(27, "unique Key values")

# Key parsing function: split into pitch class (key_num) and mode
PITCH_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

def parse_key(key_str):
    note, tonality = key_str.split(" ")
    key_num = PITCH_MAP[note]
    mode = 1 if tonality == "Maj" else 0
    return pd.Series([key_num, mode])

spotify_df[["key_num", "mode"]] = spotify_df["Key"].apply(parse_key)

# Verifying the parsed values against a 10-row sample
render_table(spotify_df[["Key", "key_num", "mode"]].head(10), figsize=(10, 6), fontsize=11)
save_fig(29, "key parsing verification")

# Renaming Positiveness to valence and displaying the updated column list
spotify_df = spotify_df.rename(columns={"Positiveness": "valence"})
render_table(pd.DataFrame(spotify_df.columns, columns=["Column"]), figsize=(6, 10), fontsize=10)
save_fig(30, "updated column list")

# Post-engineering structural verification
spotify_df.info()

# Investigating the distribution of the variance column
plt.figure(figsize=(10, 6))
sns.histplot(spotify_df["variance"], bins=40, kde=True, color="slateblue")
plt.title("Distribution of the variance column", fontsize=14, fontweight="bold")
plt.xlabel("variance")
save_fig(32, "variance column investigation")

print(spotify_df["variance"].describe())
print(f"Negative-value rows: {(spotify_df['variance'] < 0).sum()}")

# 4.3.3.2  DATA TRANSFORMATION - PHASE 2 NLP PIPELINE FOR LYRICS

# --------------

def strip_metadata(raw_text: str) -> str:
    # remove bracketed section tags, e.g. [Chorus], [Verse 1], [Bridge]
    text = re.sub(r"\[[^\]]*\]", " ", raw_text)
    # remove parenthetical ad-libs and embedded timestamps (mm:ss)
    text = re.sub(r"\(\d{1,2}:\d{2}\)", " ", text)
    # collapse residual HTML tags carried over from scraping
    text = re.sub(r"<[^>]+>", " ", text)
    # collapse repeated whitespace introduced by the substitutions above
    text = re.sub(r"\s+", " ", text).strip()
    return text

spotify_df["lyrics_clean"] = spotify_df["lyrics_raw"].astype(str).apply(strip_metadata)

# ------------------------
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [tok for tok in tokens if tok not in STOPWORDS]
    return " ".join(filtered)

spotify_df["lyrics_clean"] = spotify_df["lyrics_clean"].str.lower()
spotify_df["lyrics_clean"] = spotify_df["lyrics_clean"].apply(remove_stopwords)

# -----------------
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# Purge Instrumental Anomalies (Instrumentalness > 0.8)
spotify_df = spotify_df[~((spotify_df['Instrumentalness'] > 0.8) & 
                          (spotify_df['lyrics_clean'].str.contains(r'\[instrumental', na=False)))]

# Enforce Linguistic Uniformity (English Only)
spotify_df['is_english'] = spotify_df['lyrics_clean'].apply(is_english)
spotify_df = spotify_df[spotify_df['is_english'] == True]
spotify_df = spotify_df.drop(columns=['is_english'])

# ---------------------------
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

encoded = tokenizer(
    spotify_df["lyrics_clean"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=128,
    return_attention_mask=True,
    return_tensors="pt",
)

input_ids = encoded["input_ids"]            # shape: (n_samples, 128)
attention_mask = encoded["attention_mask"]  # shape: (n_samples, 128)

# 4.3.4  DATA NORMALIZATION  (DEAM valence/arousal -> 0-100 scale)

# Pre-normalisation descriptive statistics for the DEAM dimensions
print(deam_df[["valence_mean", "arousal_mean"]].describe())

# Applying the (X-1)/(9-1) x 100 min-max formula and extracting the median threshold
deam_df["valence_norm"] = (deam_df["valence_mean"] - 1) / (9 - 1) * 100
deam_df["arousal_norm"] = (deam_df["arousal_mean"] - 1) / (9 - 1) * 100

valence_threshold = deam_df["valence_norm"].median()
arousal_threshold = deam_df["arousal_norm"].median()
print(f"Valence threshold: {valence_threshold}")
print(f"Arousal threshold: {arousal_threshold}")

# Post-normalisation descriptive statistics
print(deam_df[["valence_norm", "arousal_norm"]].describe())

# Histogram for normalised DEAM valence
plt.figure(figsize=(8, 6))
sns.histplot(deam_df["valence_norm"], bins=30, kde=True, color="mediumpurple")
plt.axvline(valence_threshold, color="red", linestyle="--", label=f"Median = {valence_threshold}")
plt.title("Distribution of Normalised DEAM Valence", fontsize=14, fontweight="bold")
plt.legend()
save_fig(43, "valence_norm histogram")

# Histogram for normalised DEAM arousal
plt.figure(figsize=(8, 6))
sns.histplot(deam_df["arousal_norm"], bins=30, kde=True, color="teal")
plt.axvline(arousal_threshold, color="red", linestyle="--", label=f"Median = {arousal_threshold}")
plt.title("Distribution of Normalised DEAM Arousal", fontsize=14, fontweight="bold")
plt.legend()
save_fig(44, "arousal_norm histogram")

# Boxplot for normalised DEAM valence
plt.figure(figsize=(6, 8))
sns.boxplot(y=deam_df["valence_norm"], color="mediumpurple")
plt.title("Boxplot of Normalised Valence", fontsize=14, fontweight="bold")
save_fig(45, "valence_norm boxplot")

# Boxplot for normalised DEAM arousal
plt.figure(figsize=(6, 8))
sns.boxplot(y=deam_df["arousal_norm"], color="teal")
plt.title("Boxplot of Normalised Arousal", fontsize=14, fontweight="bold")
save_fig(46, "arousal_norm boxplot")

# DEAM circumplex space scatter plot
plt.figure(figsize=(9, 9))
plt.scatter(deam_df["valence_norm"], deam_df["arousal_norm"], alpha=0.5, s=15, color="darkorange")
plt.axvline(valence_threshold, color="black", linestyle="--")
plt.axhline(arousal_threshold, color="black", linestyle="--")
plt.xlabel("Normalised Valence")
plt.ylabel("Normalised Arousal")
plt.title("DEAM Annotations on Russell's Circumplex Model", fontsize=14, fontweight="bold")
save_fig(48, "DEAM circumplex scatter")

# Threshold derivation summary table
summary = pd.DataFrame({
    "Dimension": ["Valence", "Arousal"],
    "Mean (raw 1-9)": [deam_df["valence_mean"].mean(), deam_df["arousal_mean"].mean()],
    "Std (raw 1-9)": [deam_df["valence_mean"].std(), deam_df["arousal_mean"].std()],
    "Median (normalised)": [valence_threshold, arousal_threshold],
})
render_table(summary, figsize=(10, 3), fontsize=11)
save_fig(49, "threshold derivation summary table")

# 4.4.5  EMOTION LABEL APPLICATION AND COMPARISON

# Plotting the DEAM quadrant label distribution
def deam_quadrant(row):
    if row["valence_norm"] >= valence_threshold and row["arousal_norm"] >= arousal_threshold:
        return "Happy"
    elif row["valence_norm"] < valence_threshold and row["arousal_norm"] >= arousal_threshold:
        return "Angry"
    elif row["valence_norm"] >= valence_threshold and row["arousal_norm"] < arousal_threshold:
        return "Calm"
    else:
        return "Sad"

deam_df["deam_quadrant"] = deam_df.apply(deam_quadrant, axis=1)
plt.figure(figsize=(9, 6))
sns.countplot(data=deam_df, x="deam_quadrant", order=["Happy", "Angry", "Calm", "Sad"], palette=emotion_palette)
plt.title("DEAM Quadrant Label Distribution", fontsize=14, fontweight="bold")
save_fig(68, "DEAM quadrant distribution")

# Applying the 48.75 thresholds to the Spotify dataset
def assign_quadrant(row):
    if row["valence"] >= 48.75 and row["Energy"] >= 48.75:
        return "Happy"
    elif row["valence"] < 48.75 and row["Energy"] >= 48.75:
        return "Angry"
    elif row["valence"] >= 48.75 and row["Energy"] < 48.75:
        return "Calm"
    else:
        return "Sad"

spotify_df["emotion_label"] = spotify_df.apply(assign_quadrant, axis=1)
render_table(
    spotify_df[["Song-Name", "valence", "Energy", "emotion_label"]].sample(10, random_state=42),
    figsize=(12, 5), fontsize=10,
)
save_fig(70, "emotion_label sample")

# Tabulating the frequency distribution of the new emotion_label column
new_counts = spotify_df["emotion_label"].value_counts().reset_index()
new_counts.columns = ["Emotion Label", "Count"]
render_table(new_counts, figsize=(10, 5), fontsize=11)
save_fig(44, "emotion label value counts")

# Generating a bar chart of the new emotion_label distribution
plt.figure(figsize=(10, 7))
ax = sns.countplot(data=spotify_df, x="emotion_label", palette=emotion_palette, order=["Happy", "Angry", "Calm", "Sad"])
plt.title("Distribution of DEAM-Validated Emotion Labels (Spotify)", fontsize=14, fontweight="bold")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="center", xytext=(0, 8), textcoords="offset points", fontsize=11)
save_fig(45, "emotion label bar chart")

# Generating a pie chart of the new emotion_label distribution
plt.figure(figsize=(8, 8))
spotify_df["emotion_label"].value_counts().reindex(["Happy", "Angry", "Calm", "Sad"]).plot.pie(
    autopct="%1.1f%%", colors=[emotion_palette[l] for l in ["Happy", "Angry", "Calm", "Sad"]]
)
plt.title("Proportional Distribution of Emotion Labels", fontsize=14, fontweight="bold")
plt.ylabel("")
save_fig(46, "emotion label pie chart")

# Re-plotting the original six-category commercial emotion distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=spotify_df, y="emotion", order=spotify_df["emotion"].value_counts().index, color="slategray")
plt.title("Original Commercial Emotion Label Distribution", fontsize=14, fontweight="bold")
save_fig(23, "commercial emotion distribution")

# Side-by-side comparison of the original and new label distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(data=spotify_df, y="emotion", order=spotify_df["emotion"].value_counts().index,
              color="slategray", ax=axes[0])
axes[0].set_title("Original Commercial Labels (6 categories)")
sns.countplot(data=spotify_df, x="emotion_label", palette=emotion_palette,
              order=["Happy", "Angry", "Calm", "Sad"], ax=axes[1])
axes[1].set_title("Threshold-Derived Labels (4 quadrants)")
plt.tight_layout()
save_fig(48, "label distribution comparison")

# Constructing the crosstab between the original emotion labels and the new quadrants
cross_tab = pd.crosstab(spotify_df["emotion"], spotify_df["emotion_label"])
render_table(cross_tab.reset_index(), figsize=(10, 4), fontsize=10)
save_fig(50, "emotion crosstab table")

# Rendering the crosstab as a heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Original Emotion vs. Threshold-Derived Quadrant", fontsize=14, fontweight="bold")
save_fig(51, "emotion crosstab heatmap")

# 4.4.4  HISTOGRAMS AND VISUALIZATIONS  (acoustic feature distributions)

HIST_FEATURES = [
    ("valence", 86, "skyblue", "navy"),
    ("Energy", 88, "salmon", "darkred"),
    ("Danceability", 90, "lightgreen", "darkgreen"),
    ("Tempo", 92, "khaki", "olive"),
    ("Loudness", 55, "skyblue", "navy"),
    ("Speechiness", 96, "plum", "purple"),
    ("Liveness", 98, "lightcoral", "brown"),
    ("Acousticness", 100, "wheat", "chocolate"),
    ("Instrumentalness", 102, "lightsteelblue", "midnightblue"),
]

for feat, fig_num, c1, c2 in HIST_FEATURES:
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(spotify_df[feat], bins=30, kde=False, color=c1)
    plt.title(f"Histogram of {feat}", fontsize=13)
    plt.subplot(1, 2, 2)
    sns.kdeplot(spotify_df[feat], fill=True, color=c2)
    plt.title(f"KDE of {feat}", fontsize=13)
    save_fig(fig_num, f"{feat} distribution")

# Bar chart for key_num distribution
plt.figure(figsize=(11, 6))
sns.countplot(data=spotify_df, x="key_num", color="cornflowerblue", order=sorted(spotify_df["key_num"].unique()))
plt.title("Distribution of key_num (Pitch Class)", fontsize=14, fontweight="bold")
save_fig(104, "key_num distribution")

# Bar chart for mode distribution
plt.figure(figsize=(7, 6))
sns.countplot(data=spotify_df, x="mode", color="goldenrod")
plt.xticks([0, 1], ["minor", "Major"])
plt.title("Distribution of Mode", fontsize=14, fontweight="bold")
save_fig(105, "mode distribution")

# Bar chart for Genre distribution
plt.figure(figsize=(12, 8))
top_genres = spotify_df["Genre"].value_counts().head(20)
sns.barplot(x=top_genres.values, y=top_genres.index, color="mediumseagreen")
plt.title("Top 20 Genres by Frequency", fontsize=14, fontweight="bold")
save_fig(106, "Genre distribution")

# Full boxplot grid across all numerical audio features
numerical_cols = [
    "valence", "Energy", "Danceability", "Tempo", "Loudness",
    "Speechiness", "Liveness", "Acousticness", "Instrumentalness",
    "key_num", "mode",
]
fig, axes = plt.subplots(4, 3, figsize=(16, 18))
for ax, col in zip(axes.flatten(), numerical_cols):
    sns.boxplot(y=spotify_df[col], ax=ax, color="lightblue")
    ax.set_title(col)
plt.tight_layout()
save_fig(116, "boxplot grid all features")

# Computing the exact IQR-based outlier count for every numerical feature
def iqr_outlier_count(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

outlier_counts = {col: iqr_outlier_count(spotify_df[col]) for col in numerical_cols}
outlier_table = pd.DataFrame(list(outlier_counts.items()), columns=["Feature", "Outlier Count"])
outlier_table = outlier_table.sort_values("Outlier Count", ascending=False)
render_table(outlier_table, figsize=(8, 6), fontsize=11)
save_fig(118, "IQR outlier counts")

# Documenting the outlier retention decision
print("Decision: retain all IQR-flagged outliers.")
print("Rationale: extreme Instrumentalness, Acousticness, and Loudness values")
print("reflect authentic musical properties (e.g. purely instrumental or")
print("ambient/quiet tracks) rather than data-entry errors.")
print(f"Row count unchanged: {spotify_df.shape[0]}")

# One-Way ANOVA across all acoustic features by emotion class
order = ["Happy", "Angry", "Calm", "Sad"]
anova_results = []
for col in numerical_cols:
    groups = [spotify_df[spotify_df["emotion_label"] == label][col] for label in order]
    f_stat, p_val = f_oneway(*groups)
    anova_results.append({"Feature": col, "F-statistic": f_stat, "p-value": p_val})

anova_table = pd.DataFrame(anova_results).sort_values("F-statistic", ascending=False)
render_table(anova_table, figsize=(9, 6), fontsize=11)
save_fig(140, "ANOVA results table")

# Circumplex Model scatter plot with quadrant thresholds
plt.figure(figsize=(10, 10))
sns.scatterplot(data=spotify_df, x="valence", y="Energy", hue="emotion_label",
                 palette=emotion_palette, alpha=0.4, s=12)
plt.axvline(48.75, color="black", linestyle="--")
plt.axhline(48.75, color="black", linestyle="--")
plt.title("Russell's Circumplex Model (Spotify, thresholded at 48.75)", fontsize=14, fontweight="bold")
save_fig(112, "final circumplex scatter")

# 4.4.6  CORRELATION AND MULTIVARIATE ANALYSIS

# Computing and rendering the Pearson correlation heatmap
corr_matrix = spotify_df[numerical_cols].corr(method="pearson")
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Pearson Correlation Matrix (Acoustic Features)", fontsize=14, fontweight="bold")
save_fig(63, "correlation heatmap")

# Raw valence versus Energy scatter plot
plt.figure(figsize=(9, 8))
sns.scatterplot(data=spotify_df, x="valence", y="Energy", alpha=0.3, s=10, color="teal")
plt.title("valence vs. Energy (raw joint distribution)", fontsize=14, fontweight="bold")
save_fig(64, "valence vs Energy scatter")

# Multi-feature pairplot coloured by emotion_label
pair_cols = ["valence", "Energy", "Danceability", "Loudness", "Acousticness", "Tempo", "emotion_label"]
sample_df = spotify_df[pair_cols].sample(5000, random_state=42)
sns.pairplot(sample_df, hue="emotion_label", palette=emotion_palette,
             plot_kws={"alpha": 0.4, "s": 15}, diag_kind="kde")
save_fig(65, "pairplot core features")

# 4.4.7  PER-CLASS FEATURE ANALYSIS

# Tabulating per-class mean values for the core acoustic features
class_means = spotify_df.groupby("emotion_label")[
    ["valence", "Energy", "Danceability", "Loudness", "Acousticness", "Tempo"]
].mean().reindex(["Happy", "Angry", "Calm", "Sad"])
render_table(class_means.reset_index(), figsize=(12, 4), fontsize=10)
save_fig(69, "per-class mean value table")

PER_CLASS_FEATURES = [
    ("valence", 70),
    ("Energy", 71),
    ("Danceability", 72),
    ("Acousticness", 73),
    ("Loudness", 74),
    ("Tempo", 75),
]
for feat, fig_num in PER_CLASS_FEATURES:
    plt.figure(figsize=(11, 7))
    sns.boxplot(data=spotify_df, x="emotion_label", y=feat, palette=emotion_palette,
                order=["Happy", "Angry", "Calm", "Sad"])
    plt.title(f"{feat} distribution by Emotion Class", fontsize=14, fontweight="bold")
    save_fig(fig_num, f"boxplot {feat} by emotion")

# valence violin plot grouped by emotion_label
plt.figure(figsize=(11, 7))
sns.violinplot(data=spotify_df, x="emotion_label", y="valence", palette=emotion_palette,
               order=["Happy", "Angry", "Calm", "Sad"])
plt.title("valence density by Emotion Class", fontsize=14, fontweight="bold")
save_fig(76, "violin valence by emotion")

# Energy violin plot grouped by emotion_label
plt.figure(figsize=(11, 7))
sns.violinplot(data=spotify_df, x="emotion_label", y="Energy", palette=emotion_palette,
               order=["Happy", "Angry", "Calm", "Sad"])
plt.title("Energy density by Emotion Class", fontsize=14, fontweight="bold")
save_fig(77, "violin Energy by emotion")

# 4.3.5  DATA BALANCING  (class imbalance documentation + SMOTE)

# Calculating and tabulating the final emotion label counts and percentages
label_counts = spotify_df["emotion_label"].value_counts()
label_pct = (label_counts / label_counts.sum() * 100).round(2)
summary_table = pd.DataFrame({"Count": label_counts, "Percentage": label_pct})
render_table(summary_table.reset_index(), figsize=(8, 4), fontsize=11)
save_fig(79, "final class distribution table")

# Generating the final class distribution bar chart with percentage annotations
plt.figure(figsize=(11, 7))
ax = sns.countplot(data=spotify_df, x="emotion_label", palette=emotion_palette, order=order)
plt.title("Final Class Distribution (Spotify Dataset)", fontsize=14, fontweight="bold")
totals = spotify_df["emotion_label"].value_counts()
for p in ax.patches:
    height = p.get_height()
    pct = height / totals.sum() * 100
    ax.annotate(f"{int(height)}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
save_fig(80, "final class distribution bar chart")

# The originally documented (superseded) class imbalance handling strategy
plt.figure(figsize=(11, 5))
plt.text(
    0.5, 0.5,
    "Class Imbalance Handling Strategy:\n\n"
    "1. Use class_weight='balanced'\n   during model training.\n"
    "2. Evaluate using F1-Score and\n   Confusion Matrix (not just Accuracy).\n\n"
    "No Synthetic Oversampling (SMOTE) to maintain the authenticity\nof musical feature relationships.",
    ha="center", va="center", fontsize=15, fontweight="bold",
    bbox=dict(facecolor="#dff0f8", edgecolor="red", boxstyle="round,pad=1"),
)
plt.axis("off")
save_fig(81, "imbalance handling strategy")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

feature_cols = [
    "Danceability", "Tempo", "Loudness",
    "Speechiness", "Liveness", "Acousticness", "Instrumentalness",
    "key_num", "mode",
]
X = spotify_df[feature_cols]
y = spotify_df["emotion_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Before SMOTE:", Counter(y_train))

smote = SMOTE(k_neighbors=5, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("After SMOTE:", Counter(y_train_bal))

# 4.4.3 (cont.) / 4.4.8  FINAL DESCRIPTIVE STATISTICS AND DATASET SUMMARY

# Generating the final descriptive statistics table for the cleaned, engineered dataset
final_desc = spotify_df.describe()
render_table(final_desc.reset_index(), figsize=(20, 4), fontsize=8)
save_fig(154, "final descriptive statistics")

# Final .info() summary of the fully processed dataset
spotify_df.info()

# Verifying the final dataset shape
print(f"Final dataset shape: {spotify_df.shape}")

# Final null verification across the complete processed dataset
final_nulls = spotify_df.isnull().sum().reset_index()
final_nulls.columns = ["Column", "Null Count"]
render_table(final_nulls, figsize=(10, 10), fontsize=10)
save_fig(84, "final null verification")

# Previewing the first five records of the final processed dataset
render_table(spotify_df.head(), figsize=(20, 3), fontsize=8)
save_fig(86, "final dataset head")

# Exporting the final processed dataset to disk
spotify_df.to_csv("spotify_prepared_final.csv", index=False)
plt.figure(figsize=(10, 4))
plt.text(
    0.5, 0.5,
    f"Dataset saved successfully:\n'spotify_prepared_final.csv'\n\n"
    f"Rows: {len(spotify_df)}\nColumns: {len(spotify_df.columns)}",
    ha="center", va="center", fontsize=14, fontweight="bold",
    bbox=dict(facecolor="lightgreen", alpha=0.3),
)
plt.axis("off")
save_fig(87, "dataset save confirmation")

# Sub-module: stage_1_2_data_preparation.py

"""
=============================================================================
STAGE 1 & 2: Data Loading, Leakage Prevention, Splitting, Scaling & Encoding
=============================================================================
Purpose: Verify dataset integrity, remove data leakage, encode labels,
         perform stratified train-test split, and scale features.
         
This script does NOT train any models. It only prepares and validates data.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
OUTPUT_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 1: DATA LOADING & LEAKAGE PREVENTION                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("STAGE 1: DATA LOADING & LEAKAGE PREVENTION")
print("=" * 70)

# ── Step 1.1: Load raw dataset ──
print("\n[Step 1.1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  ✓ Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 1.2: Inspect all columns ──
print("\n[Step 1.2] All columns in dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>2}. {col} (dtype: {df[col].dtype})")

# ── Step 1.3: Check for missing values ──
print("\n[Step 1.3] Missing values check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found.")
else:
    print("  ⚠ Missing values detected:")
    print(missing[missing > 0])

# ── Step 1.4: Check for duplicates ──
print("\n[Step 1.4] Duplicate rows check:")
dup_count = df.duplicated().sum()
print(f"  {'✓ No duplicates found.' if dup_count == 0 else f'⚠ {dup_count} duplicate rows detected!'}")

# ── Step 1.5: Inspect the target variable ──
print("\n[Step 1.5] Target variable (emotion_label) distribution:")
target_dist = df['emotion_label'].value_counts()
target_pct = df['emotion_label'].value_counts(normalize=True) * 100
for label in target_dist.index:
    print(f"  {label:>8}: {target_dist[label]:>6} samples  ({target_pct[label]:.2f}%)")
print(f"  Total: {target_dist.sum()} samples")

# ── Step 1.6: Identify and separate columns ──
print("\n[Step 1.6] Column classification:")

# Metadata: Non-predictive columns (identifiers, text, categorical not useful for ML)
metadata_cols = ['artist', 'Song-Name', 'emotion', 'variance', 'Genre', 'Release Date', 'Key']
print(f"  Metadata (to drop):  {metadata_cols}")

# Leakage: These were used to DERIVE emotion_label, so including them = circular logic
leakage_cols = ['valence', 'Energy']
print(f"  Leakage (to drop):   {leakage_cols}")

# Non-acoustic: These are not inherent acoustic properties of the music signal
non_acoustic_cols = ['Explicit', 'Popularity']
print(f"  Non-acoustic (to drop): {non_acoustic_cols}")

# Target
target_col = 'emotion_label'
print(f"  Target:              [{target_col}]")

# Remaining features for ML
all_drop = metadata_cols + leakage_cols + non_acoustic_cols + [target_col]
feature_cols = [col for col in df.columns if col not in all_drop]
print(f"\n  ✓ Features for ML ({len(feature_cols)}): {feature_cols}")

# ── Step 1.7: Apply the drops ──
print("\n[Step 1.7] Dropping non-feature columns...")
X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"  ✓ X shape: {X.shape}  (should be {df.shape[0]} × {len(feature_cols)})")
print(f"  ✓ y shape: {y.shape}  (should be {df.shape[0]},)")

# ── Step 1.8: Verify no leakage columns remain ──
print("\n[Step 1.8] LEAKAGE VERIFICATION:")
for col in leakage_cols:
    if col in X.columns:
        print(f"  ✗ CRITICAL ERROR: '{col}' is still in the feature set!")
    else:
        print(f"  ✓ '{col}' successfully removed from feature set.")

# ── Step 1.9: Feature data types and ranges ──
print("\n[Step 1.9] Feature statistics (pre-scaling):")
print(X.describe().round(2).to_string())

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 2: ENCODING, SPLITTING & SCALING                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("STAGE 2: ENCODING, SPLITTING & SCALING")
print("=" * 70)

# ── Step 2.1: Label Encoding ──
print("\n[Step 2.1] Encoding target labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("  Class mapping:")
for cls, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"    {cls:>8} → {code}")

# ── Step 2.2: Stratified Train-Test Split ──
print("\n[Step 2.2] Performing 80/20 stratified split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"  ✓ X_train: {X_train.shape}")
print(f"  ✓ X_test:  {X_test.shape}")
print(f"  ✓ y_train: {y_train.shape}")
print(f"  ✓ y_test:  {y_test.shape}")

# ── Step 2.3: Verify stratification preserved class ratios ──
print("\n[Step 2.3] Stratification verification (class % should be ~equal):")
print(f"  {'Class':>8} | {'Full Data':>10} | {'Train Set':>10} | {'Test Set':>10}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

full_counts = pd.Series(y_encoded).value_counts(normalize=True).sort_index()
train_counts = pd.Series(y_train).value_counts(normalize=True).sort_index()
test_counts = pd.Series(y_test).value_counts(normalize=True).sort_index()

for idx in range(len(le.classes_)):
    cls_name = le.classes_[idx]
    full_pct = full_counts.get(idx, 0) * 100
    train_pct = train_counts.get(idx, 0) * 100
    test_pct = test_counts.get(idx, 0) * 100
    print(f"  {cls_name:>8} | {full_pct:>9.2f}% | {train_pct:>9.2f}% | {test_pct:>9.2f}%")

# ── Step 2.4: Feature Scaling ──
print("\n[Step 2.4] Applying StandardScaler...")
print("  ⚠ IMPORTANT: Scaler is fit ONLY on X_train to prevent data leakage.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ X_train_scaled shape: {X_train_scaled.shape}")
print(f"  ✓ X_test_scaled shape:  {X_test_scaled.shape}")

# ── Step 2.5: Verify scaling correctness ──
print("\n[Step 2.5] Scaling verification (train set should have mean≈0, std≈1):")
train_means = X_train_scaled.mean(axis=0)
train_stds = X_train_scaled.std(axis=0)
print(f"  {'Feature':>20} | {'Mean':>8} | {'Std':>8}")
print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")
for i, col in enumerate(feature_cols):
    print(f"  {col:>20} | {train_means[i]:>8.4f} | {train_stds[i]:>8.4f}")

# ── Step 2.6: Save all artifacts ──
print("\n[Step 2.6] Saving artifacts...")
import joblib

joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# Save scaled data for model training in Stage 3
np.save(os.path.join(OUTPUT_DIR, "X_train_scaled.npy"), X_train_scaled)
np.save(os.path.join(OUTPUT_DIR, "X_test_scaled.npy"), X_test_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# Save feature names for later use
with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w") as f:
    f.write("\n".join(feature_cols))

print(f"  ✓ Saved label_encoder.pkl")
print(f"  ✓ Saved scaler.pkl")
print(f"  ✓ Saved X_train_scaled.npy, X_test_scaled.npy")
print(f"  ✓ Saved y_train.npy, y_test.npy")
print(f"  ✓ Saved feature_names.txt")
print(f"\n  All artifacts saved to: {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("STAGE 1 & 2 COMPLETE — Ready for Stage 3 (Model Training)")
print("=" * 70)
