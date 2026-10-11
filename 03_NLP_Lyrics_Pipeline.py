# Sub-module: stage_5a_lyrics_scraper.py

"""
=============================================================================
STAGE 5A: LYRICS SCRAPER (GENIUS API)
=============================================================================
Purpose: Extract a balanced subset of 20,000 songs and scrape their lyrics
         using the Genius API to build the dataset for the NLP model.

INSTRUCTIONS BEFORE RUNNING:
1. Go to https://genius.com/api-clients and log in / create a free account.
2. Click "New API Client".
3. Name it "FYP_MER" (Website URL can be anything, e.g., http://localhost).
4. Click "Save".
5. Copy the "Client Access Token" and paste it in the GENIUS_ACCESS_TOKEN 
   variable below.
=============================================================================
"""

import sys
import io
import os
import time
import pandas as pd
import lyricsgenius
from requests.exceptions import Timeout, ReadTimeout

# Fix encoding issues for terminal printing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
# 1. PASTE YOUR ACCESS TOKEN HERE
# ══════════════════════════════════════════════════════════════════════════════
GENIUS_ACCESS_TOKEN = "uaRhJd6QePTQjCHm-Av-N30H1hz7zNiHKfKpaVHiuCgzVYm4MbflzL6cH766WAdW"

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
OUTPUT_FILE  = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_lyrics_20k.csv"
SAMPLES_PER_CLASS = 5000  # 5,000 * 4 emotions = 20,000 total

print("=" * 70)
print("STAGE 5A: GENIUS LYRICS SCRAPER")
print("=" * 70)

if GENIUS_ACCESS_TOKEN == "PASTE_YOUR_TOKEN_HERE":
    print("\n[ERROR] You forgot to add your Genius API Token!")
    print("Please open 'stage_5a_lyrics_scraper.py' and paste your token at line 23.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CREATE THE 20K SUBSET
# ══════════════════════════════════════════════════════════════════════════════
# Check if we already have a partial file from a previous run to resume from
if os.path.exists(OUTPUT_FILE):
    print(f"\n[Loading] Found existing progress file: {OUTPUT_FILE}")
    subset_df = pd.read_csv(OUTPUT_FILE)
    print(f"  ✓ Loaded {len(subset_df)} rows.")
else:
    print("\n[Creating Subset] Loading original dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # We only want songs that actually have an artist and track name
    df = df.dropna(subset=['artist', 'Song-Name'])
    
    # Randomly sample 5,000 from each emotion class
    subset_df = df.groupby('emotion_label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=42)
    ).reset_index(drop=True)
    
    # Add an empty column for lyrics
    subset_df['lyrics'] = None
    
    # Save the initial empty-lyrics file
    subset_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Created new subset with {len(subset_df)} rows. Saved to {OUTPUT_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. INITIALIZE GENIUS API
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Connecting] Initializing Genius API...")
genius = lyricsgenius.Genius(
    GENIUS_ACCESS_TOKEN, 
    skip_non_songs=True, 
    excluded_terms=["(Remix)", "(Live)"],
    remove_section_headers=True,
    timeout=15,    # 15 seconds before giving up on a slow song
    retries=3      # Try 3 times if it times out
)
# Turn off verbose mode so it doesn't flood the terminal
genius.verbose = False 

# ══════════════════════════════════════════════════════════════════════════════
# 4. START SCRAPING (WITH AUTO-SAVE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Scraping] Starting the download process...")
print("  (This will take many hours. You can press Ctrl+C to stop at any time.")
print("   The script auto-saves every 50 songs, so you can safely resume later.)\n")

total_songs = len(subset_df)
# Find how many we have already processed (where lyrics is not NA and not empty string)
# Actually, it's safer to track which ones we've attempted. We'll use a new column 'scraped_attempted'
if 'scrape_attempted' not in subset_df.columns:
    subset_df['scrape_attempted'] = False

success_count = 0
fail_count = 0
start_time = time.time()

for index, row in subset_df.iterrows():
    # Skip if we already attempted to scrape this song
    if row['scrape_attempted'] == True:
        continue
        
    track = str(row['Song-Name']).strip()
    artist = str(row['artist']).strip()
    
    try:
        # Search Genius
        song = genius.search_song(track, artist)
        
        if song is not None and song.lyrics:
            subset_df.at[index, 'lyrics'] = song.lyrics
            success_count += 1
            status = "✓ FOUND"
        else:
            subset_df.at[index, 'lyrics'] = "NOT_FOUND"
            fail_count += 1
            status = "✗ NOT FOUND"
            
    except Exception as e:
        subset_df.at[index, 'lyrics'] = "ERROR"
        fail_count += 1
        status = f"⚠ ERROR ({str(e)[:30]})"
        
    # Mark as attempted
    subset_df.at[index, 'scrape_attempted'] = True
    
    # Print progress
    print(f"[{index+1}/{total_songs}] {status}: {track[:30]} - {artist[:20]}")
    
    # Auto-save every 50 songs
    if (index + 1) % 50 == 0:
        subset_df.to_csv(OUTPUT_FILE, index=False)
        elapsed = time.time() - start_time
        print(f"  --> AUTO-SAVED (Took {elapsed:.1f}s for last batch. Success: {success_count}, Fails: {fail_count})")
        start_time = time.time()
        
    # Sleep to respect rate limits
    time.sleep(0.5)

# Final save when loop finishes
subset_df.to_csv(OUTPUT_FILE, index=False)
print("\n" + "=" * 70)
print("SCRAPING COMPLETE!")
print(f"Total Found: {len(subset_df[subset_df['lyrics'].notna() & (subset_df['lyrics'] != 'NOT_FOUND') & (subset_df['lyrics'] != 'ERROR')])}")
print("=" * 70)

# Sub-module: stage_5a_v2_balanced_scraper.py

"""
=============================================================================
STAGE 5A v2: BALANCED LYRICS SCRAPER (GENIUS API)
=============================================================================
Purpose: Re-scrape lyrics with SHUFFLED order so all 4 emotion classes
         get equal coverage before the API rate limit kicks in.
         
         We already have ~4,920 Angry and ~4,891 Calm lyrics.
         We need Happy and Sad lyrics. This script focuses on those.
=============================================================================
"""

import sys
import io
import os
import time
import pandas as pd
import lyricsgenius
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── API Token ──
GENIUS_ACCESS_TOKEN = "uaRhJd6QePTQjCHm-Av-N30H1hz7zNiHKfKpaVHiuCgzVYm4MbflzL6cH766WAdW"

# ── Paths ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
OUTPUT_FILE  = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_lyrics_happy_sad.csv"

print("=" * 70)
print("STAGE 5A v2: SCRAPING HAPPY & SAD LYRICS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# Check if we have a partial file to resume from
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists(OUTPUT_FILE):
    print(f"\n[Resuming] Found existing progress file: {OUTPUT_FILE}")
    subset_df = pd.read_csv(OUTPUT_FILE)
    print(f"  ✓ Loaded {len(subset_df)} rows.")
else:
    print("\n[Creating Subset] Loading original dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['artist', 'Song-Name'])
    
    # Only get Happy and Sad songs (we already have Angry and Calm)
    happy_df = df[df['emotion_label'] == 'Happy'].sample(n=5000, random_state=42)
    sad_df   = df[df['emotion_label'] == 'Sad'].sample(n=5000, random_state=42)
    
    subset_df = pd.concat([happy_df, sad_df], ignore_index=True)
    
    # SHUFFLE so we get interleaved Happy/Sad instead of sequential
    subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    subset_df['lyrics'] = None
    subset_df['scrape_attempted'] = False
    
    subset_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Created shuffled subset: {len(subset_df)} rows (Happy + Sad)")
    print(f"    Happy: {len(happy_df)}, Sad: {len(sad_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# Initialize Genius
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Connecting] Initializing Genius API...")
genius = lyricsgenius.Genius(
    GENIUS_ACCESS_TOKEN,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
    remove_section_headers=True,
    timeout=15,
    retries=3
)
genius.verbose = False

# ══════════════════════════════════════════════════════════════════════════════
# Scrape with error detection & early stop
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Scraping] Starting download (shuffled Happy & Sad)...")
print("  Auto-saves every 50 songs. Safe to Ctrl+C at any time.\n")

total_songs = len(subset_df)
success_count = 0
fail_count = 0
consecutive_errors = 0  # Track API ban detection
start_time = time.time()

for index, row in subset_df.iterrows():
    if row['scrape_attempted'] == True:
        continue
    
    track = str(row['Song-Name']).strip()
    artist = str(row['artist']).strip()
    
    try:
        song = genius.search_song(track, artist)
        
        if song is not None and song.lyrics:
            subset_df.at[index, 'lyrics'] = song.lyrics
            success_count += 1
            status = "✓ FOUND"
            consecutive_errors = 0  # Reset error counter
        else:
            subset_df.at[index, 'lyrics'] = "NOT_FOUND"
            fail_count += 1
            status = "✗ NOT FOUND"
            consecutive_errors = 0
            
    except Exception as e:
        subset_df.at[index, 'lyrics'] = "ERROR"
        fail_count += 1
        status = f"⚠ ERROR ({str(e)[:30]})"
        consecutive_errors += 1
        
        # If we get 20 consecutive errors, the API has banned us
        if consecutive_errors >= 20:
            print(f"\n[API BAN DETECTED] 20 consecutive errors. Stopping to preserve data.")
            subset_df.to_csv(OUTPUT_FILE, index=False)
            break
    
    subset_df.at[index, 'scrape_attempted'] = True
    print(f"[{index+1}/{total_songs}] {status}: {track[:30]} - {artist[:20]}")
    
    if (index + 1) % 50 == 0:
        subset_df.to_csv(OUTPUT_FILE, index=False)
        elapsed = time.time() - start_time
        print(f"  --> AUTO-SAVED (Took {elapsed:.1f}s. Success: {success_count}, Fails: {fail_count})")
        start_time = time.time()
    
    time.sleep(0.5)

# Final save
subset_df.to_csv(OUTPUT_FILE, index=False)

# Report
valid = subset_df[subset_df['lyrics'].notna() & (subset_df['lyrics'] != 'NOT_FOUND') & (subset_df['lyrics'] != 'ERROR')]
print("\n" + "=" * 70)
print("SCRAPING COMPLETE!")
print(f"  Total Found: {len(valid)}")
print(f"  Per Class:")
print(valid['emotion_label'].value_counts().to_string())
print("=" * 70)

# Sub-module: stage_5b_colab_nlp_training.py

"""
=============================================================================
STAGE 5B: DISTILBERT NLP FINE-TUNING (GOOGLE COLAB)
=============================================================================
INSTRUCTIONS:
1. Go to https://colab.research.google.com/ and create a "New Notebook"
2. Go to Runtime > Change runtime type > Select "T4 GPU" (Important!)
3. Upload 'spotify_lyrics_balanced_final.csv' to the Colab files pane
4. Copy and paste all the code below into a single cell and press Run.
=============================================================================
"""

# 1. Install required libraries (HuggingFace Transformers, Datasets, Evaluate)
!pip install -q transformers datasets evaluate accelerate scikit-learn

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.model_selection import train_test_split

print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))

# 2. Load the Balanced Dataset
print("\nLoading dataset...")
df = pd.read_csv('spotify_lyrics_balanced_final.csv')

# Drop any accidental NaNs in lyrics
df = df.dropna(subset=['lyrics'])

# Map emotions to integers (Required for PyTorch)
emotion_map = {'Angry': 0, 'Calm': 1, 'Happy': 2, 'Sad': 3}
df['label'] = df['emotion_label'].map(emotion_map)

# Take a subset if memory crashes, but T4 GPU should handle 16k fine.
print("Class distribution:")
print(df['emotion_label'].value_counts())

# Split into Train and Test (80/20)
train_df, test_df = train_test_split(df[['lyrics', 'label']], test_size=0.2, random_state=42, stratify=df['label'])

# Convert pandas dataframes to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

# 3. Load DistilBERT Tokenizer
print("\nLoading DistilBERT Tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Truncate lyrics to 512 tokens (BERT's maximum context window)
    return tokenizer(examples["lyrics"], padding="max_length", truncation=True, max_length=512)

print("Tokenizing datasets (this takes a minute)...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 4. Load Pre-trained Model
print("\nLoading DistilBERT Model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# 5. Define Evaluation Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

# 6. Training Arguments (IMPROVED: 10 epochs with early stopping)
training_args = TrainingArguments(
    output_dir="./distilbert_lyrics_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1.5e-5,            # Slightly lower LR for longer training stability
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,             # 10 epochs (early stopping will pick the best)
    weight_decay=0.01,
    warmup_ratio=0.1,               # Warm up for first 10% of training steps
    load_best_model_at_end=True,
    metric_for_best_model="f1",     # Pick the checkpoint with best F1 score
    greater_is_better=True,
    push_to_hub=False,
)

# 7. Initialize Trainer with Early Stopping
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop if no improvement for 3 epochs
)

# 8. Train!
print("\nStarting Fine-Tuning...")
trainer.train()

# 9. Final Evaluation
print("\nFinal Evaluation on Test Set:")
eval_results = trainer.evaluate()
print(eval_results)

# 10. Save the Final Model to Disk
print("\nSaving model...")
model.save_pretrained("./final_distilbert_lyrics")
tokenizer.save_pretrained("./final_distilbert_lyrics")

# Zip the model so it's easy to download from Colab to your Windows Laptop
!zip -r final_distilbert_lyrics.zip ./final_distilbert_lyrics
print("\nDONE! You can now right-click 'final_distilbert_lyrics.zip' in the Colab files pane and click 'Download'!")
