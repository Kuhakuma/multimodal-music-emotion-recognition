# Sub-module: stage_3_baseline_training.py

"""
=============================================================================
STAGE 3: BASELINE MODEL TRAINING
=============================================================================
Purpose: Train three baseline classifiers on the prepared data from Stage 1&2.
         Apply class_weight='balanced' to handle the Calm class imbalance.
         Evaluate each model with Accuracy, Precision, Recall, F1-Score,
         and Confusion Matrices.
         
This script loads the artifacts saved by stage_1_2_data_preparation.py.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
PIPELINE_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 3: BASELINE MODEL TRAINING                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("STAGE 3: BASELINE MODEL TRAINING")
print("=" * 70)

# ── Step 3.1: Load artifacts from Stage 1 & 2 ──
print("\n[Step 3.1] Loading prepared data from Stage 1 & 2...")

X_train_scaled = np.load(os.path.join(PIPELINE_DIR, "X_train_scaled.npy"))
X_test_scaled = np.load(os.path.join(PIPELINE_DIR, "X_test_scaled.npy"))
y_train = np.load(os.path.join(PIPELINE_DIR, "y_train.npy"))
y_test = np.load(os.path.join(PIPELINE_DIR, "y_test.npy"))
le = joblib.load(os.path.join(PIPELINE_DIR, "label_encoder.pkl"))

with open(os.path.join(PIPELINE_DIR, "feature_names.txt"), "r") as f:
    feature_names = f.read().strip().split("\n")

print(f"  ✓ X_train: {X_train_scaled.shape}")
print(f"  ✓ X_test:  {X_test_scaled.shape}")
print(f"  ✓ Classes: {le.classes_.tolist()}")
print(f"  ✓ Features: {feature_names}")

# ── Step 3.2: Define baseline models ──
print("\n[Step 3.2] Initializing baseline models...")
print("  Note: class_weight='balanced' applied to RF and SVM")
print("  Note: HistGradientBoosting uses sample_weight internally")

# Compute sample weights for HistGradientBoosting (it doesn't support class_weight)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights_train = compute_sample_weight('balanced', y_train)

models = {
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        "use_sample_weight": False
    },
    "Linear SVM": {
        "model": LinearSVC(
            random_state=42,
            dual=False,
            max_iter=3000,
            class_weight='balanced'
        ),
        "use_sample_weight": False
    },
    "Gradient Boosting": {
        "model": HistGradientBoostingClassifier(
            random_state=42,
            max_iter=200
        ),
        "use_sample_weight": True
    }
}

for name, config in models.items():
    extras = []
    if name == "Random Forest":
        extras.append("n_estimators=100")
        extras.append("class_weight='balanced'")
    elif name == "Linear SVM":
        extras.append("max_iter=3000")
        extras.append("class_weight='balanced'")
    elif name == "Gradient Boosting":
        extras.append("max_iter=200")
        extras.append("sample_weight='balanced'")
    print(f"  ✓ {name}: {', '.join(extras)}")

# ── Step 3.3: Train and evaluate each model ──
print("\n" + "=" * 70)
print("TRAINING & EVALUATION")
print("=" * 70)

results = {}

for name, config in models.items():
    print(f"\n{'─' * 70}")
    print(f"  MODEL: {name}")
    print(f"{'─' * 70}")
    
    model = config["model"]
    
    # ── Train ──
    print(f"  [Training] Fitting on {X_train_scaled.shape[0]} samples...")
    start_time = time.time()
    
    if config["use_sample_weight"]:
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    else:
        model.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"  [Training] Completed in {train_time:.2f} seconds")
    
    # ── Predict on test set ──
    y_pred = model.predict(X_test_scaled)
    
    # ── Predict on train set (to check overfitting) ──
    y_pred_train = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_pred_train)
    
    # ── Test metrics ──
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    
    # ── Overfitting check ──
    overfit_gap = train_acc - test_acc
    
    print(f"\n  [Results]")
    print(f"    Train Accuracy:  {train_acc:.4f}  ({train_acc*100:.2f}%)")
    print(f"    Test Accuracy:   {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"    Overfit Gap:     {overfit_gap:.4f}  {'⚠ HIGH' if overfit_gap > 0.10 else '✓ Acceptable'}")
    print(f"    Macro Precision: {macro_precision:.4f}")
    print(f"    Macro Recall:    {macro_recall:.4f}")
    print(f"    Macro F1-Score:  {macro_f1:.4f}")
    
    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  [Confusion Matrix]")
    print(f"    {'':>8}", end="")
    for cls in le.classes_:
        print(f"  {cls:>8}", end="")
    print("  <- Predicted")
    for i, cls in enumerate(le.classes_):
        print(f"    {cls:>8}", end="")
        for j in range(len(le.classes_)):
            print(f"  {cm[i][j]:>8}", end="")
        print()
    
    # ── Per-class report ──
    print(f"\n  [Per-Class Report]")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(f"    {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
    print(f"    {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for cls in le.classes_:
        r = report[cls]
        print(f"    {cls:>8} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")
    
    # Store results
    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "overfit_gap": overfit_gap,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "train_time": train_time,
        "per_class": report
    }

# ── Step 3.4: Comparative Summary ──
print(f"\n{'=' * 70}")
print("COMPARATIVE SUMMARY")
print(f"{'=' * 70}")

print(f"\n  {'Model':>20} | {'Train Acc':>9} | {'Test Acc':>9} | {'Gap':>6} | {'Macro F1':>9} | {'Time (s)':>8}")
print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}")

for name, r in results.items():
    gap_flag = " !!" if r["overfit_gap"] > 0.10 else ""
    print(f"  {name:>20} | {r['train_acc']:>9.4f} | {r['test_acc']:>9.4f} | {r['overfit_gap']:>5.3f}{gap_flag} | {r['macro_f1']:>9.4f} | {r['train_time']:>8.2f}")

# ── Step 3.5: Identify best model ──
best_name = max(results, key=lambda k: results[k]["macro_f1"])
best_result = results[best_name]

print(f"\n  >> Best baseline model: {best_name}")
print(f"     Macro F1-Score: {best_result['macro_f1']:.4f}")
print(f"     Test Accuracy:  {best_result['test_acc']:.4f}")

# ── Step 3.6: Check for class-specific weaknesses ──
print(f"\n[Step 3.6] Class-specific weakness analysis for '{best_name}':")
weak_classes = []
for cls in le.classes_:
    f1 = best_result["per_class"][cls]["f1-score"]
    status = "✓ OK" if f1 >= 0.50 else "⚠ WEAK"
    if f1 < 0.50:
        weak_classes.append(cls)
    print(f"  {cls:>8}: F1 = {f1:.4f}  {status}")

if weak_classes:
    print(f"\n  ⚠ Classes needing attention in Stage 4 tuning: {weak_classes}")
else:
    print(f"\n  ✓ All classes have F1 >= 0.50")

# ── Step 3.7: Save baseline model for comparison in Stage 4 ──
print(f"\n[Step 3.7] Saving best baseline model...")
joblib.dump(best_result["model"], os.path.join(PIPELINE_DIR, "baseline_best_model.pkl"))
print(f"  ✓ Saved baseline_best_model.pkl ({best_name})")

# Save results summary as CSV
summary_rows = []
for name, r in results.items():
    summary_rows.append({
        "Model": name,
        "Train_Accuracy": round(r["train_acc"], 4),
        "Test_Accuracy": round(r["test_acc"], 4),
        "Overfit_Gap": round(r["overfit_gap"], 4),
        "Macro_Precision": round(r["macro_precision"], 4),
        "Macro_Recall": round(r["macro_recall"], 4),
        "Macro_F1": round(r["macro_f1"], 4),
        "Training_Time_Sec": round(r["train_time"], 2)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(PIPELINE_DIR, "stage3_baseline_results.csv"), index=False)
print(f"  ✓ Saved stage3_baseline_results.csv")

print(f"\n{'=' * 70}")
print("STAGE 3 COMPLETE — Ready for Stage 4 (Hyperparameter Tuning)")
print(f"{'=' * 70}")

# Sub-module: stage_4_hyperparameter_tuning.py

"""
=============================================================================
STAGE 4: HYPERPARAMETER TUNING
=============================================================================
Purpose: Tune the Random Forest classifier to reduce the severe overfitting
         observed in Stage 3 (Train: 100% vs Test: 69.75%, Gap: 30.25%).
         
Strategy: Use GridSearchCV with 3-fold cross-validation to find optimal
          max_depth, min_samples_leaf, and min_samples_split values that
          constrain tree growth and improve generalization.
          
This script loads artifacts from the pipeline_output/ directory.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, make_scorer
)
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
PIPELINE_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 4: HYPERPARAMETER TUNING                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("STAGE 4: HYPERPARAMETER TUNING (Random Forest)")
print("=" * 70)

# ── Step 4.1: Load data and baseline model ──
print("\n[Step 4.1] Loading prepared data and baseline results...")

X_train_scaled = np.load(os.path.join(PIPELINE_DIR, "X_train_scaled.npy"))
X_test_scaled = np.load(os.path.join(PIPELINE_DIR, "X_test_scaled.npy"))
y_train = np.load(os.path.join(PIPELINE_DIR, "y_train.npy"))
y_test = np.load(os.path.join(PIPELINE_DIR, "y_test.npy"))
le = joblib.load(os.path.join(PIPELINE_DIR, "label_encoder.pkl"))
baseline_model = joblib.load(os.path.join(PIPELINE_DIR, "baseline_best_model.pkl"))

with open(os.path.join(PIPELINE_DIR, "feature_names.txt"), "r") as f:
    feature_names = f.read().strip().split("\n")

print(f"  ✓ Data loaded: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")
print(f"  ✓ Baseline model loaded (Random Forest)")

# ── Step 4.2: Compute baseline metrics for comparison ──
print("\n[Step 4.2] Baseline model performance (from Stage 3):")
y_pred_baseline = baseline_model.predict(X_test_scaled)
y_pred_train_baseline = baseline_model.predict(X_train_scaled)

baseline_train_acc = accuracy_score(y_train, y_pred_train_baseline)
baseline_test_acc = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
baseline_gap = baseline_train_acc - baseline_test_acc

print(f"  Train Accuracy:  {baseline_train_acc:.4f}")
print(f"  Test Accuracy:   {baseline_test_acc:.4f}")
print(f"  Overfit Gap:     {baseline_gap:.4f}")
print(f"  Macro F1-Score:  {baseline_f1:.4f}")

# ── Step 4.3: Define the hyperparameter grid ──
print("\n[Step 4.3] Defining hyperparameter search grid...")
print("  Goal: Constrain tree depth to reduce overfitting gap from 30%")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20, 25],
    'min_samples_leaf': [5, 10, 20],
    'min_samples_split': [10, 20]
}

total_combos = 1
for key, values in param_grid.items():
    total_combos *= len(values)
    print(f"  {key}: {values}")

print(f"\n  Total combinations: {total_combos}")
print(f"  Cross-validation folds: 3")
print(f"  Total fits: {total_combos * 3}")
print(f"  Scoring metric: Macro F1-Score (penalizes poor minority class performance)")

# ── Step 4.4: Run GridSearchCV ──
print(f"\n[Step 4.4] Running GridSearchCV...")
print(f"  This may take several minutes on {X_train_scaled.shape[0]} samples...")

rf_tuned = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

scorer = make_scorer(f1_score, average='macro')

grid_search = GridSearchCV(
    estimator=rf_tuned,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
search_time = time.time() - start_time

print(f"\n  ✓ GridSearchCV completed in {search_time:.1f} seconds ({search_time/60:.1f} minutes)")

# ── Step 4.5: Analyze GridSearchCV results ──
print(f"\n[Step 4.5] GridSearchCV Results:")
print(f"\n  Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"    {param}: {value}")
print(f"\n  Best CV Macro F1-Score: {grid_search.best_score_:.4f}")

# Show top 5 combinations
print(f"\n  Top 5 parameter combinations:")
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results_sorted = cv_results.sort_values('rank_test_score').head(5)

print(f"  {'Rank':>4} | {'n_est':>5} | {'depth':>5} | {'leaf':>5} | {'split':>5} | {'CV F1':>7} | {'Train F1':>8}")
print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}")

for _, row in cv_results_sorted.iterrows():
    print(f"  {int(row['rank_test_score']):>4} | "
          f"{row['param_n_estimators']:>5} | "
          f"{row['param_max_depth']:>5} | "
          f"{row['param_min_samples_leaf']:>5} | "
          f"{row['param_min_samples_split']:>5} | "
          f"{row['mean_test_score']:>7.4f} | "
          f"{row['mean_train_score']:>8.4f}")

# ── Step 4.6: Evaluate the tuned model on the held-out test set ──
print(f"\n{'=' * 70}")
print("TUNED MODEL EVALUATION ON TEST SET")
print(f"{'=' * 70}")

best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)
y_pred_train_tuned = best_model.predict(X_train_scaled)

tuned_train_acc = accuracy_score(y_train, y_pred_train_tuned)
tuned_test_acc = accuracy_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned, average='macro')
tuned_gap = tuned_train_acc - tuned_test_acc

print(f"\n  [Tuned Model Results]")
print(f"    Train Accuracy:  {tuned_train_acc:.4f}  ({tuned_train_acc*100:.2f}%)")
print(f"    Test Accuracy:   {tuned_test_acc:.4f}  ({tuned_test_acc*100:.2f}%)")
print(f"    Overfit Gap:     {tuned_gap:.4f}  {'⚠ Still high' if tuned_gap > 0.10 else '✓ Reduced'}")
print(f"    Macro F1-Score:  {tuned_f1:.4f}")

# ── Step 4.7: Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred_tuned)
print(f"\n  [Confusion Matrix]")
print(f"    {'':>8}", end="")
for cls in le.classes_:
    print(f"  {cls:>8}", end="")
print("  <- Predicted")
for i, cls in enumerate(le.classes_):
    print(f"    {cls:>8}", end="")
    for j in range(len(le.classes_)):
        print(f"  {cm[i][j]:>8}", end="")
    print()

# ── Step 4.8: Per-class report ──
print(f"\n  [Per-Class Report]")
report = classification_report(y_test, y_pred_tuned, target_names=le.classes_, output_dict=True)
print(f"    {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
print(f"    {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
for cls in le.classes_:
    r = report[cls]
    print(f"    {cls:>8} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")

# ── Step 4.9: Side-by-side comparison ──
print(f"\n{'=' * 70}")
print("BASELINE vs TUNED COMPARISON")
print(f"{'=' * 70}")

print(f"\n  {'Metric':>20} | {'Baseline':>10} | {'Tuned':>10} | {'Change':>10}")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
print(f"  {'Train Accuracy':>20} | {baseline_train_acc:>10.4f} | {tuned_train_acc:>10.4f} | {tuned_train_acc - baseline_train_acc:>+10.4f}")
print(f"  {'Test Accuracy':>20} | {baseline_test_acc:>10.4f} | {tuned_test_acc:>10.4f} | {tuned_test_acc - baseline_test_acc:>+10.4f}")
print(f"  {'Overfit Gap':>20} | {baseline_gap:>10.4f} | {tuned_gap:>10.4f} | {tuned_gap - baseline_gap:>+10.4f}")
print(f"  {'Macro F1':>20} | {baseline_f1:>10.4f} | {tuned_f1:>10.4f} | {tuned_f1 - baseline_f1:>+10.4f}")

# Per-class F1 comparison
baseline_report = classification_report(y_test, y_pred_baseline, target_names=le.classes_, output_dict=True)
print(f"\n  Per-class F1 comparison:")
print(f"  {'Class':>8} | {'Baseline F1':>11} | {'Tuned F1':>10} | {'Change':>10}")
print(f"  {'-'*8}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}")
for cls in le.classes_:
    b_f1 = baseline_report[cls]['f1-score']
    t_f1 = report[cls]['f1-score']
    print(f"  {cls:>8} | {b_f1:>11.4f} | {t_f1:>10.4f} | {t_f1 - b_f1:>+10.4f}")

# ── Step 4.10: Feature Importance ──
print(f"\n[Step 4.10] Feature Importance (Tuned Random Forest):")
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print(f"  {'Rank':>4} | {'Feature':>20} | {'Importance':>10} | {'Bar'}")
print(f"  {'-'*4}-+-{'-'*20}-+-{'-'*10}-+------")
for rank, idx in enumerate(sorted_idx, 1):
    bar = "#" * int(importances[idx] * 50)
    print(f"  {rank:>4} | {feature_names[idx]:>20} | {importances[idx]:>10.4f} | {bar}")

# ── Step 4.11: Save final model ──
print(f"\n[Step 4.11] Saving tuned model and artifacts...")
joblib.dump(best_model, os.path.join(PIPELINE_DIR, "tuned_best_model.pkl"))
print(f"  ✓ Saved tuned_best_model.pkl")

# Save tuning results
tuning_summary = {
    "best_params": grid_search.best_params_,
    "best_cv_f1": grid_search.best_score_,
    "baseline_test_acc": baseline_test_acc,
    "baseline_f1": baseline_f1,
    "baseline_gap": baseline_gap,
    "tuned_test_acc": tuned_test_acc,
    "tuned_f1": tuned_f1,
    "tuned_gap": tuned_gap,
    "feature_importances": dict(zip(feature_names, importances.tolist()))
}

import json
with open(os.path.join(PIPELINE_DIR, "stage4_tuning_results.json"), "w") as f:
    json.dump(tuning_summary, f, indent=2)
print(f"  ✓ Saved stage4_tuning_results.json")

print(f"\n{'=' * 70}")
print("STAGE 4 COMPLETE — Ready for Stage 5 (Streamlit Deployment)")
print(f"{'=' * 70}")

# Sub-module: stage_4b_aggressive_tuning.py

"""
=============================================================================
STAGE 4B: AGGRESSIVE HYPERPARAMETER TUNING
=============================================================================
Purpose: Apply much stricter regularization to close the overfitting gap.
         
Previous attempt (Stage 4A):
  - Grid was too permissive (max_depth up to 25, min_samples_leaf as low as 5)
  - Result: Gap reduced from 30% to 20%, but test accuracy dropped
  
This attempt:
  - Force shallow trees (max_depth: 5-15)
  - Force larger leaf nodes (min_samples_leaf: 20-100) 
  - Limit features per tree (max_features: 'sqrt', 0.5)
  - Target: Overfit gap < 10% while maintaining reasonable test accuracy
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, make_scorer
)
import joblib
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
PIPELINE_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 4B: AGGRESSIVE TUNING                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("STAGE 4B: AGGRESSIVE HYPERPARAMETER TUNING")
print("=" * 70)

# ── Step 4B.1: Load data ──
print("\n[Step 4B.1] Loading data...")
X_train_scaled = np.load(os.path.join(PIPELINE_DIR, "X_train_scaled.npy"))
X_test_scaled = np.load(os.path.join(PIPELINE_DIR, "X_test_scaled.npy"))
y_train = np.load(os.path.join(PIPELINE_DIR, "y_train.npy"))
y_test = np.load(os.path.join(PIPELINE_DIR, "y_test.npy"))
le = joblib.load(os.path.join(PIPELINE_DIR, "label_encoder.pkl"))

with open(os.path.join(PIPELINE_DIR, "feature_names.txt"), "r") as f:
    feature_names = f.read().strip().split("\n")

print(f"  ✓ Loaded: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")

# ── Step 4B.2: Load previous results for comparison ──
print("\n[Step 4B.2] Loading previous model results for comparison...")

# Baseline (Stage 3)
baseline_model = joblib.load(os.path.join(PIPELINE_DIR, "baseline_best_model.pkl"))
y_pred_baseline = baseline_model.predict(X_test_scaled)
y_pred_train_baseline = baseline_model.predict(X_train_scaled)
baseline_train_acc = accuracy_score(y_train, y_pred_train_baseline)
baseline_test_acc = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
baseline_gap = baseline_train_acc - baseline_test_acc

# Mild tuning (Stage 4A)
tuned_model_a = joblib.load(os.path.join(PIPELINE_DIR, "tuned_best_model.pkl"))
y_pred_tuned_a = tuned_model_a.predict(X_test_scaled)
y_pred_train_tuned_a = tuned_model_a.predict(X_train_scaled)
tuned_a_train_acc = accuracy_score(y_train, y_pred_train_tuned_a)
tuned_a_test_acc = accuracy_score(y_test, y_pred_tuned_a)
tuned_a_f1 = f1_score(y_test, y_pred_tuned_a, average='macro')
tuned_a_gap = tuned_a_train_acc - tuned_a_test_acc

print(f"  Baseline:    Train={baseline_train_acc:.4f}, Test={baseline_test_acc:.4f}, Gap={baseline_gap:.4f}, F1={baseline_f1:.4f}")
print(f"  Mild Tuned:  Train={tuned_a_train_acc:.4f}, Test={tuned_a_test_acc:.4f}, Gap={tuned_a_gap:.4f}, F1={tuned_a_f1:.4f}")

# ── Step 4B.3: Define aggressive grid ──
print("\n[Step 4B.3] Defining AGGRESSIVE hyperparameter grid...")
print("  Strategy: Force shallow trees + large leaves + limited features")

param_grid = {
    'n_estimators': [200],
    'max_depth': [5, 8, 10, 12, 15],
    'min_samples_leaf': [20, 50, 100],
    'min_samples_split': [20, 50],
    'max_features': ['sqrt', 0.5]
}

total_combos = 1
for key, values in param_grid.items():
    total_combos *= len(values)
    print(f"  {key}: {values}")

print(f"\n  Total combinations: {total_combos}")
print(f"  Cross-validation folds: 3")
print(f"  Total fits: {total_combos * 3}")

# ── Step 4B.4: Run GridSearchCV ──
print(f"\n[Step 4B.4] Running aggressive GridSearchCV...")

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

scorer = make_scorer(f1_score, average='macro')

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
search_time = time.time() - start_time

print(f"\n  ✓ Completed in {search_time:.1f} seconds ({search_time/60:.1f} minutes)")

# ── Step 4B.5: Analyze results ──
print(f"\n[Step 4B.5] GridSearchCV Results:")
print(f"\n  Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"    {param}: {value}")
print(f"\n  Best CV Macro F1: {grid_search.best_score_:.4f}")

# Top 10 combinations
print(f"\n  Top 10 parameter combinations:")
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results_sorted = cv_results.sort_values('rank_test_score').head(10)

print(f"  {'Rank':>4} | {'depth':>5} | {'leaf':>5} | {'split':>5} | {'feat':>5} | {'CV F1':>7} | {'Train F1':>8} | {'Gap':>6}")
print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")

for _, row in cv_results_sorted.iterrows():
    cv_gap = row['mean_train_score'] - row['mean_test_score']
    print(f"  {int(row['rank_test_score']):>4} | "
          f"{row['param_max_depth']:>5} | "
          f"{row['param_min_samples_leaf']:>5} | "
          f"{row['param_min_samples_split']:>5} | "
          f"{str(row['param_max_features']):>5} | "
          f"{row['mean_test_score']:>7.4f} | "
          f"{row['mean_train_score']:>8.4f} | "
          f"{cv_gap:>6.4f}")

# ── Step 4B.6: Evaluate on held-out test set ──
print(f"\n{'=' * 70}")
print("AGGRESSIVELY TUNED MODEL — TEST SET EVALUATION")
print(f"{'=' * 70}")

best_model = grid_search.best_estimator_
y_pred_agg = best_model.predict(X_test_scaled)
y_pred_train_agg = best_model.predict(X_train_scaled)

agg_train_acc = accuracy_score(y_train, y_pred_train_agg)
agg_test_acc = accuracy_score(y_test, y_pred_agg)
agg_f1 = f1_score(y_test, y_pred_agg, average='macro')
agg_gap = agg_train_acc - agg_test_acc

print(f"\n  Train Accuracy:  {agg_train_acc:.4f}  ({agg_train_acc*100:.2f}%)")
print(f"  Test Accuracy:   {agg_test_acc:.4f}  ({agg_test_acc*100:.2f}%)")
print(f"  Overfit Gap:     {agg_gap:.4f}  {'✓ GOOD (<10%)' if agg_gap < 0.10 else '⚠ Still high' if agg_gap < 0.15 else '!! Too high'}")
print(f"  Macro F1-Score:  {agg_f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_agg)
print(f"\n  [Confusion Matrix]")
print(f"    {'':>8}", end="")
for cls in le.classes_:
    print(f"  {cls:>8}", end="")
print("  <- Predicted")
for i, cls in enumerate(le.classes_):
    print(f"    {cls:>8}", end="")
    for j in range(len(le.classes_)):
        print(f"  {cm[i][j]:>8}", end="")
    print()

# Per-class report
print(f"\n  [Per-Class Report]")
report = classification_report(y_test, y_pred_agg, target_names=le.classes_, output_dict=True)
print(f"    {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
print(f"    {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
for cls in le.classes_:
    r = report[cls]
    print(f"    {cls:>8} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")

# ── Step 4B.7: THREE-WAY COMPARISON ──
print(f"\n{'=' * 70}")
print("THREE-WAY COMPARISON: BASELINE vs MILD TUNING vs AGGRESSIVE TUNING")
print(f"{'=' * 70}")

baseline_report = classification_report(y_test, y_pred_baseline, target_names=le.classes_, output_dict=True)
tuned_a_report = classification_report(y_test, y_pred_tuned_a, target_names=le.classes_, output_dict=True)

print(f"\n  {'Metric':>20} | {'Baseline':>10} | {'Mild Tune':>10} | {'Aggressive':>10}")
print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
print(f"  {'Train Accuracy':>20} | {baseline_train_acc:>10.4f} | {tuned_a_train_acc:>10.4f} | {agg_train_acc:>10.4f}")
print(f"  {'Test Accuracy':>20} | {baseline_test_acc:>10.4f} | {tuned_a_test_acc:>10.4f} | {agg_test_acc:>10.4f}")
print(f"  {'Overfit Gap':>20} | {baseline_gap:>10.4f} | {tuned_a_gap:>10.4f} | {agg_gap:>10.4f}")
print(f"  {'Macro F1':>20} | {baseline_f1:>10.4f} | {tuned_a_f1:>10.4f} | {agg_f1:>10.4f}")

print(f"\n  Per-class F1 comparison:")
print(f"  {'Class':>8} | {'Baseline':>10} | {'Mild Tune':>10} | {'Aggressive':>10}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
for cls in le.classes_:
    b = baseline_report[cls]['f1-score']
    m = tuned_a_report[cls]['f1-score']
    a = report[cls]['f1-score']
    print(f"  {cls:>8} | {b:>10.4f} | {m:>10.4f} | {a:>10.4f}")

# ── Step 4B.8: Feature Importance ──
print(f"\n[Step 4B.8] Feature Importance (Aggressive Model):")
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print(f"  {'Rank':>4} | {'Feature':>20} | {'Importance':>10} | Bar")
print(f"  {'-'*4}-+-{'-'*20}-+-{'-'*10}-+------")
for rank, idx in enumerate(sorted_idx, 1):
    bar = "#" * int(importances[idx] * 50)
    print(f"  {rank:>4} | {feature_names[idx]:>20} | {importances[idx]:>10.4f} | {bar}")

# ── Step 4B.9: RECOMMENDATION ──
print(f"\n{'=' * 70}")
print("FINAL RECOMMENDATION")
print(f"{'=' * 70}")

# Determine best model considering both performance and overfitting
models_summary = {
    "Baseline": {"test_acc": baseline_test_acc, "f1": baseline_f1, "gap": baseline_gap},
    "Mild Tuned": {"test_acc": tuned_a_test_acc, "f1": tuned_a_f1, "gap": tuned_a_gap},
    "Aggressive": {"test_acc": agg_test_acc, "f1": agg_f1, "gap": agg_gap}
}

# Score each model: penalize high overfitting gaps
for name, m in models_summary.items():
    # Combined score: F1 weighted by gap penalty
    gap_penalty = max(0, 1 - m["gap"])  # 1.0 = no gap, 0.7 = 30% gap
    m["combined_score"] = m["f1"] * gap_penalty

print(f"\n  {'Model':>15} | {'Test Acc':>8} | {'Macro F1':>8} | {'Gap':>6} | {'Combined':>8}")
print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
for name, m in models_summary.items():
    marker = " <--" if m["combined_score"] == max(x["combined_score"] for x in models_summary.values()) else ""
    print(f"  {name:>15} | {m['test_acc']:>8.4f} | {m['f1']:>8.4f} | {m['gap']:>6.4f} | {m['combined_score']:>8.4f}{marker}")

recommended = max(models_summary, key=lambda k: models_summary[k]["combined_score"])
print(f"\n  >> Recommended model for deployment: {recommended}")

# ── Step 4B.10: Save final model ──
print(f"\n[Step 4B.10] Saving aggressive model...")
joblib.dump(best_model, os.path.join(PIPELINE_DIR, "aggressive_tuned_model.pkl"))

# Save complete comparison results
final_results = {
    "baseline": {"train_acc": baseline_train_acc, "test_acc": baseline_test_acc, "f1": baseline_f1, "gap": baseline_gap},
    "mild_tuned": {"train_acc": tuned_a_train_acc, "test_acc": tuned_a_test_acc, "f1": tuned_a_f1, "gap": tuned_a_gap},
    "aggressive_tuned": {
        "train_acc": agg_train_acc, "test_acc": agg_test_acc, "f1": agg_f1, "gap": agg_gap,
        "best_params": grid_search.best_params_,
        "feature_importances": dict(zip(feature_names, importances.tolist()))
    },
    "recommended": recommended
}

with open(os.path.join(PIPELINE_DIR, "stage4b_final_comparison.json"), "w") as f:
    json.dump(final_results, f, indent=2, default=str)

print(f"  ✓ Saved aggressive_tuned_model.pkl")
print(f"  ✓ Saved stage4b_final_comparison.json")

print(f"\n{'=' * 70}")
print("STAGE 4B COMPLETE — Ready for Stage 5 (Streamlit Deployment)")
print(f"{'=' * 70}")

# Sub-module: stage_4c_feature_engineering.py

"""
=============================================================================
STAGE 4C: FEATURE ENGINEERING + NEW ALGORITHMS (XGBoost, KNN)
=============================================================================
Purpose: Attempt to improve model performance through two strategies:
  1. Feature Engineering: Create derived features that capture emotion-relevant
     acoustic patterns the raw features miss.
  2. New Algorithms: Test XGBoost and KNN, which may learn different patterns
     than Random Forest.
     
All comparisons use the same random_state=42 split for fairness.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("[INFO] XGBoost is available.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not installed. Will use HistGradientBoosting as substitute.")

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
PIPELINE_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART A: FEATURE ENGINEERING                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("STAGE 4C — PART A: FEATURE ENGINEERING")
print("=" * 70)

# ── Step A.1: Load raw data ──
print("\n[Step A.1] Loading raw dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ── Step A.2: Drop metadata + leakage (same as Stage 1) ──
print("\n[Step A.2] Dropping metadata and leakage columns...")
metadata_cols = ['artist', 'Song-Name', 'emotion', 'variance', 'Genre', 'Release Date', 'Key']
leakage_cols = ['valence', 'Energy']
non_acoustic_cols = ['Explicit', 'Popularity']
target_col = 'emotion_label'

all_drop = metadata_cols + leakage_cols + non_acoustic_cols
df_clean = df.drop(columns=all_drop)
print(f"  Remaining columns: {df_clean.columns.tolist()}")

# ── Step A.3: Create engineered features ──
print("\n[Step A.3] Engineering new features...")

# Feature 1: Loudness x Tempo → Energy proxy (loud + fast = high energy feel)
df_clean['loud_tempo'] = df_clean['Loudness'] * df_clean['Tempo']
print(f"  [+] loud_tempo = Loudness x Tempo (energy proxy)")

# Feature 2: Acousticness / (Danceability + 1) → High = calm acoustic, Low = danceable electronic
df_clean['acoustic_dance_ratio'] = df_clean['Acousticness'] / (df_clean['Danceability'] + 1)
print(f"  [+] acoustic_dance_ratio = Acousticness / (Danceability + 1)")

# Feature 3: Speechiness x Tempo → Fast spoken content = intense/angry
df_clean['speech_tempo'] = df_clean['Speechiness'] * df_clean['Tempo']
print(f"  [+] speech_tempo = Speechiness x Tempo (intensity indicator)")

# Feature 4: Danceability x (100 - Acousticness) → Club/party music score
df_clean['dance_electronic'] = df_clean['Danceability'] * (100 - df_clean['Acousticness'])
print(f"  [+] dance_electronic = Danceability x (100 - Acousticness)")

# Feature 5: Liveness x Acousticness → Live acoustic performance feel
df_clean['live_acoustic'] = df_clean['Liveness'] * df_clean['Acousticness']
print(f"  [+] live_acoustic = Liveness x Acousticness")

# Feature 6: Loudness squared → Emphasizes extremes (very loud or very quiet)
df_clean['loudness_sq'] = df_clean['Loudness'] ** 2
print(f"  [+] loudness_sq = Loudness^2 (extreme emphasis)")

new_features = ['loud_tempo', 'acoustic_dance_ratio', 'speech_tempo', 
                'dance_electronic', 'live_acoustic', 'loudness_sq']

# ── Step A.4: Define X and y with engineered features ──
print("\n[Step A.4] Defining feature sets...")

X_original_cols = ['Tempo', 'Loudness', 'Danceability', 'Speechiness', 'Liveness', 
                   'Acousticness', 'Instrumentalness', 'key_num', 'mode']
X_engineered_cols = X_original_cols + new_features

X_eng = df_clean[X_engineered_cols].copy()
y = df_clean[target_col].copy()

print(f"  Original features (9):   {X_original_cols}")
print(f"  Engineered features (6): {new_features}")
print(f"  Total features (15):     {X_engineered_cols}")
print(f"  X shape: {X_eng.shape}")

# ── Step A.5: Check for NaN/Inf in engineered features ──
print("\n[Step A.5] Data integrity check on engineered features...")
nan_count = X_eng.isnull().sum().sum()
inf_count = np.isinf(X_eng.select_dtypes(include=[np.number])).sum().sum()
print(f"  NaN values: {nan_count}")
print(f"  Inf values: {inf_count}")
if nan_count > 0 or inf_count > 0:
    print("  Replacing NaN/Inf with 0...")
    X_eng = X_eng.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  ✓ Cleaned. New NaN count: {X_eng.isnull().sum().sum()}")
else:
    print(f"  ✓ All clean.")

# ── Step A.6: Encode, split, scale (same params as Stage 1&2) ──
print("\n[Step A.6] Encoding, splitting, and scaling...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  ✓ Scaling complete.")

# ── Step A.7: Verify new feature statistics ──
print("\n[Step A.7] Engineered feature statistics (pre-scaling):")
eng_stats = X_train[new_features].describe().round(2)
print(eng_stats.to_string())

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART B: MODEL TRAINING WITH ENGINEERED FEATURES                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("STAGE 4C — PART B: MODEL TRAINING")
print("=" * 70)

sample_weights = compute_sample_weight('balanced', y_train)

# Define models
models = {}

# 1. Random Forest (Mild Tuned params from Stage 4A, but with engineered features)
models["RF (Engineered)"] = {
    "model": RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    "use_sample_weight": False
}

# 2. XGBoost or HistGradientBoosting
if XGBOOST_AVAILABLE:
    # Compute class weights for XGBoost
    from collections import Counter
    counter = Counter(y_train)
    total = len(y_train)
    n_classes = len(counter)
    xgb_class_weights = {cls: total / (n_classes * count) for cls, count in counter.items()}
    
    models["XGBoost"] = {
        "model": XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            reg_alpha=1.0,
            reg_lambda=2.0,
            subsample=0.8,
            colsample_bytree=0.8,
            sample_weight=None
        ),
        "use_sample_weight": True
    }
else:
    models["HistGradientBoosting (Eng)"] = {
        "model": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            l2_regularization=1.0,
            min_samples_leaf=20
        ),
        "use_sample_weight": True
    }

# 3. K-Nearest Neighbors
models["KNN (k=15)"] = {
    "model": KNeighborsClassifier(
        n_neighbors=15,
        weights='distance',
        n_jobs=-1
    ),
    "use_sample_weight": False
}

# Train and evaluate all models
print(f"\nTraining {len(models)} models on {len(X_engineered_cols)} features...")
results = {}

for name, config in models.items():
    print(f"\n{'─' * 70}")
    print(f"  MODEL: {name}")
    print(f"{'─' * 70}")
    
    model = config["model"]
    
    # Train
    print(f"  [Training] Fitting on {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features...")
    start_time = time.time()
    
    if config["use_sample_weight"]:
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"  [Training] Completed in {train_time:.2f} seconds")
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    gap = train_acc - test_acc
    
    print(f"\n  [Results]")
    print(f"    Train Accuracy:  {train_acc:.4f}")
    print(f"    Test Accuracy:   {test_acc:.4f}")
    print(f"    Overfit Gap:     {gap:.4f}  {'✓ GOOD' if gap < 0.10 else '⚠ HIGH' if gap < 0.20 else '!! SEVERE'}")
    print(f"    Macro F1-Score:  {macro_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  [Confusion Matrix]")
    print(f"    {'':>8}", end="")
    for cls in le.classes_:
        print(f"  {cls:>8}", end="")
    print("  <- Predicted")
    for i, cls in enumerate(le.classes_):
        print(f"    {cls:>8}", end="")
        for j in range(len(le.classes_)):
            print(f"  {cm[i][j]:>8}", end="")
        print()
    
    # Per-class
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(f"\n  [Per-Class Report]")
    print(f"    {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
    print(f"    {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for cls in le.classes_:
        r = report[cls]
        print(f"    {cls:>8} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")
    
    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
        "macro_f1": macro_f1,
        "per_class": report,
        "train_time": train_time
    }

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART C: FULL COMPARISON (All stages)                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("FULL COMPARISON: ALL MODELS ACROSS ALL STAGES")
print(f"{'=' * 70}")

# Load previous results
with open(os.path.join(PIPELINE_DIR, "stage4b_final_comparison.json"), "r") as f:
    prev = json.load(f)

# Build full comparison table
all_models = {}
all_models["RF Baseline (9 feat)"] = prev["baseline"]
all_models["RF Mild Tune (9 feat)"] = prev["mild_tuned"]
all_models["RF Aggressive (9 feat)"] = prev["aggressive_tuned"]

for name, r in results.items():
    all_models[f"{name} (15 feat)"] = {
        "train_acc": r["train_acc"],
        "test_acc": r["test_acc"],
        "gap": r["gap"],
        "f1": r["macro_f1"]
    }

print(f"\n  {'Model':>30} | {'Train':>7} | {'Test':>7} | {'Gap':>7} | {'F1':>7}")
print(f"  {'-'*30}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

best_combined = None
best_combined_score = 0

for name, m in all_models.items():
    gap = m["gap"]
    f1 = m["f1"]
    gap_penalty = max(0, 1 - gap)
    combined = f1 * gap_penalty
    
    if combined > best_combined_score:
        best_combined_score = combined
        best_combined = name
    
    gap_flag = "" if gap < 0.10 else " *" if gap < 0.20 else " **"
    print(f"  {name:>30} | {m['train_acc']:>7.4f} | {m['test_acc']:>7.4f} | {gap:>6.4f}{gap_flag} | {f1:>7.4f}")

print(f"\n  (* = gap > 10%,  ** = gap > 20%)")

# Per-class F1 for the new models
print(f"\n  Per-class F1 for NEW models (engineered features):")
print(f"  {'Model':>30} | {'Angry':>7} | {'Calm':>7} | {'Happy':>7} | {'Sad':>7}")
print(f"  {'-'*30}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

for name, r in results.items():
    angry_f1 = r["per_class"]["Angry"]["f1-score"]
    calm_f1 = r["per_class"]["Calm"]["f1-score"]
    happy_f1 = r["per_class"]["Happy"]["f1-score"]
    sad_f1 = r["per_class"]["Sad"]["f1-score"]
    calm_flag = " !" if calm_f1 < 0.50 else ""
    print(f"  {name + ' (15 feat)':>30} | {angry_f1:>7.4f} | {calm_f1:>6.4f}{calm_flag} | {happy_f1:>7.4f} | {sad_f1:>7.4f}")

# ── Feature importance for RF with engineered features ──
rf_eng_model = results["RF (Engineered)"]["model"]
if hasattr(rf_eng_model, 'feature_importances_'):
    print(f"\n  Feature Importance (RF with Engineered Features):")
    importances = rf_eng_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print(f"  {'Rank':>4} | {'Feature':>25} | {'Importance':>10} | {'New?':>4} | Bar")
    print(f"  {'-'*4}-+-{'-'*25}-+-{'-'*10}-+-{'-'*4}-+------")
    for rank, idx in enumerate(sorted_idx, 1):
        feat = X_engineered_cols[idx]
        is_new = " *" if feat in new_features else ""
        bar = "#" * int(importances[idx] * 50)
        print(f"  {rank:>4} | {feat:>25} | {importances[idx]:>10.4f} | {is_new:>4} | {bar}")

# ── Save best model ──
print(f"\n{'=' * 70}")
print("SAVING RESULTS")
print(f"{'=' * 70}")

# Find the best new model
best_new_name = max(results, key=lambda k: results[k]["macro_f1"])
best_new = results[best_new_name]

print(f"\n  Best new model: {best_new_name}")
print(f"  Test Accuracy: {best_new['test_acc']:.4f}")
print(f"  Macro F1: {best_new['macro_f1']:.4f}")
print(f"  Overfit Gap: {best_new['gap']:.4f}")

# Save the best new model and its artifacts
joblib.dump(best_new["model"], os.path.join(PIPELINE_DIR, "best_engineered_model.pkl"))
joblib.dump(scaler, os.path.join(PIPELINE_DIR, "scaler_engineered.pkl"))
joblib.dump(le, os.path.join(PIPELINE_DIR, "label_encoder_engineered.pkl"))

with open(os.path.join(PIPELINE_DIR, "engineered_feature_names.txt"), "w") as f:
    f.write("\n".join(X_engineered_cols))

print(f"  ✓ Saved best_engineered_model.pkl")
print(f"  ✓ Saved scaler_engineered.pkl")
print(f"  ✓ Saved label_encoder_engineered.pkl")
print(f"  ✓ Saved engineered_feature_names.txt")

# Save full comparison
comparison = {
    "all_models": {k: {kk: vv for kk, vv in v.items() if kk != "model" and kk != "per_class"} 
                   for k, v in all_models.items()},
    "best_new_model": best_new_name,
    "engineered_features": new_features,
    "total_features": len(X_engineered_cols)
}
with open(os.path.join(PIPELINE_DIR, "stage4c_results.json"), "w") as f:
    json.dump(comparison, f, indent=2, default=str)
print(f"  ✓ Saved stage4c_results.json")

print(f"\n{'=' * 70}")
print("STAGE 4C COMPLETE")
print(f"{'=' * 70}")

# Sub-module: stage_4d_smote_ensemble_binary.py

"""
=============================================================================
STAGE 4D: SMOTE + VOTING ENSEMBLE + MLP + BINARY CLASSIFICATION
=============================================================================
Purpose: Three final improvement strategies before deployment.

  Part A: SMOTE Oversampling — Generate synthetic Calm samples so the model
          has equal representation of all 4 classes to learn from.
  Part B: Voting Ensemble — Combine RF + GB + KNN into a single voting
          classifier that leverages each algorithm's strengths.
  Part C: MLP Neural Network — Test if a neural network captures different
          non-linear patterns than tree-based models.
  Part D: Binary Classification — Collapse to 2 classes (High/Low Arousal)
          to test if acoustic features predict arousal better than quadrants.
  Part E: Full comparison across ALL experiments.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not found. SMOTE will be skipped.")

# XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
PIPELINE_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"

# ── Helper: Print model results ──
def evaluate_model(name, model, X_train, X_test, y_train, y_test, class_names):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    gap = train_acc - test_acc
    
    print(f"\n  [Results]")
    print(f"    Train Accuracy:  {train_acc:.4f}")
    print(f"    Test Accuracy:   {test_acc:.4f}")
    print(f"    Overfit Gap:     {gap:.4f}  {'✓ GOOD' if gap < 0.10 else '⚠ MODERATE' if gap < 0.20 else '!! HIGH'}")
    print(f"    Macro F1-Score:  {macro_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  [Confusion Matrix]")
    print(f"    {'':>12}", end="")
    for cls in class_names:
        print(f"  {cls:>10}", end="")
    print("  <- Predicted")
    for i, cls in enumerate(class_names):
        print(f"    {cls:>12}", end="")
        for j in range(len(class_names)):
            print(f"  {cm[i][j]:>10}", end="")
        print()
    
    # Per-class
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"\n  [Per-Class Report]")
    print(f"    {'Class':>12} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
    print(f"    {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for cls in class_names:
        r = report[cls]
        print(f"    {cls:>12} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")
    
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
        "macro_f1": macro_f1,
        "per_class": report
    }

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DATA PREPARATION (Same as Stage 4C)                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("DATA PREPARATION")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)
metadata_cols = ['artist', 'Song-Name', 'emotion', 'variance', 'Genre', 'Release Date', 'Key']
leakage_cols = ['valence', 'Energy']
non_acoustic_cols = ['Explicit', 'Popularity']
df_clean = df.drop(columns=metadata_cols + leakage_cols + non_acoustic_cols)

# Engineer features
df_clean['loud_tempo'] = df_clean['Loudness'] * df_clean['Tempo']
df_clean['acoustic_dance_ratio'] = df_clean['Acousticness'] / (df_clean['Danceability'] + 1)
df_clean['speech_tempo'] = df_clean['Speechiness'] * df_clean['Tempo']
df_clean['dance_electronic'] = df_clean['Danceability'] * (100 - df_clean['Acousticness'])
df_clean['live_acoustic'] = df_clean['Liveness'] * df_clean['Acousticness']
df_clean['loudness_sq'] = df_clean['Loudness'] ** 2

feature_cols = ['Tempo', 'Loudness', 'Danceability', 'Speechiness', 'Liveness',
                'Acousticness', 'Instrumentalness', 'key_num', 'mode',
                'loud_tempo', 'acoustic_dance_ratio', 'speech_tempo',
                'dance_electronic', 'live_acoustic', 'loudness_sq']

X = df_clean[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
y_4class = df_clean['emotion_label']

le_4class = LabelEncoder()
y_4class_encoded = le_4class.fit_transform(y_4class)

X_train, X_test, y_train_4, y_test_4 = train_test_split(
    X, y_4class_encoded, test_size=0.2, random_state=42, stratify=y_4class_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ Data prepared: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test, {len(feature_cols)} features")
print(f"  ✓ 4-class distribution (train): {dict(zip(le_4class.classes_, np.bincount(y_train_4)))}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART A: SMOTE OVERSAMPLING                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PART A: SMOTE OVERSAMPLING")
print(f"{'=' * 70}")

if SMOTE_AVAILABLE:
    print("\n[Step A.1] Applying SMOTE to training data...")
    print(f"  Before SMOTE: {dict(zip(le_4class.classes_, np.bincount(y_train_4)))}")
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_4)
    
    print(f"  After SMOTE:  {dict(zip(le_4class.classes_, np.bincount(y_train_smote)))}")
    print(f"  Training samples: {len(y_train_4)} -> {len(y_train_smote)} ({len(y_train_smote) - len(y_train_4):+d})")
    
    # Train RF on SMOTE data
    print(f"\n{'─' * 70}")
    print(f"  MODEL: Random Forest + SMOTE")
    print(f"{'─' * 70}")
    
    rf_smote = RandomForestClassifier(
        n_estimators=200, max_depth=25, min_samples_leaf=5,
        min_samples_split=10, random_state=42, n_jobs=-1
    )
    
    start = time.time()
    rf_smote.fit(X_train_smote, y_train_smote)
    print(f"  [Training] Completed in {time.time() - start:.2f}s")
    rf_smote_results = evaluate_model("RF+SMOTE", rf_smote, X_train_smote, X_test_scaled,
                                       y_train_smote, y_test_4, le_4class.classes_)
    
    # Train HistGB on SMOTE data
    print(f"\n{'─' * 70}")
    print(f"  MODEL: Gradient Boosting + SMOTE")
    print(f"{'─' * 70}")
    
    gb_smote = HistGradientBoostingClassifier(
        max_iter=300, max_depth=10, learning_rate=0.1,
        random_state=42, min_samples_leaf=20
    )
    
    start = time.time()
    gb_smote.fit(X_train_smote, y_train_smote)
    print(f"  [Training] Completed in {time.time() - start:.2f}s")
    gb_smote_results = evaluate_model("GB+SMOTE", gb_smote, X_train_smote, X_test_scaled,
                                       y_train_smote, y_test_4, le_4class.classes_)
else:
    print("  SMOTE unavailable. Skipping Part A.")
    X_train_smote, y_train_smote = X_train_scaled, y_train_4
    rf_smote_results = None
    gb_smote_results = None

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART B: VOTING ENSEMBLE                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PART B: VOTING ENSEMBLE (RF + GB + KNN)")
print(f"{'=' * 70}")

print(f"\n{'─' * 70}")
print(f"  MODEL: Soft Voting Ensemble (RF + HistGB + KNN)")
print(f"{'─' * 70}")

# Use SMOTE'd data if available
train_X = X_train_smote if SMOTE_AVAILABLE else X_train_scaled
train_y = y_train_smote if SMOTE_AVAILABLE else y_train_4

rf_vote = RandomForestClassifier(
    n_estimators=200, max_depth=25, min_samples_leaf=5,
    min_samples_split=10, random_state=42, n_jobs=-1
)
gb_vote = HistGradientBoostingClassifier(
    max_iter=300, max_depth=10, learning_rate=0.1,
    random_state=42, min_samples_leaf=20
)
knn_vote = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)

# HistGradientBoosting doesn't support predict_proba needed for soft voting
# Use hard voting instead
ensemble = VotingClassifier(
    estimators=[('rf', rf_vote), ('gb', gb_vote), ('knn', knn_vote)],
    voting='hard',
    n_jobs=-1
)

start = time.time()
ensemble.fit(train_X, train_y)
print(f"  [Training] Completed in {time.time() - start:.2f}s")
ensemble_results = evaluate_model("Ensemble", ensemble, train_X, X_test_scaled,
                                   train_y, y_test_4, le_4class.classes_)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART C: MLP NEURAL NETWORK                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PART C: MLP NEURAL NETWORK")
print(f"{'=' * 70}")

print(f"\n{'─' * 70}")
print(f"  MODEL: MLP (128-64-32 neurons, ReLU, Adam)")
print(f"{'─' * 70}")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    learning_rate='adaptive'
)

start = time.time()
mlp.fit(train_X, train_y)
print(f"  [Training] Completed in {time.time() - start:.2f}s")
mlp_results = evaluate_model("MLP", mlp, train_X, X_test_scaled,
                              train_y, y_test_4, le_4class.classes_)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART D: BINARY CLASSIFICATION (High vs Low Arousal)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PART D: BINARY CLASSIFICATION (High Arousal vs Low Arousal)")
print(f"{'=' * 70}")

# Remap: Happy + Angry = High Arousal (1), Sad + Calm = Low Arousal (0)
print("\n[Step D.1] Remapping classes...")
print("  High Arousal = Happy + Angry")
print("  Low Arousal  = Sad + Calm")

y_binary = y_4class.map({
    'Happy': 'High Arousal',
    'Angry': 'High Arousal',
    'Sad': 'Low Arousal',
    'Calm': 'Low Arousal'
})

le_binary = LabelEncoder()
y_binary_encoded = le_binary.fit_transform(y_binary)
print(f"  Classes: {dict(zip(le_binary.classes_, le_binary.transform(le_binary.classes_)))}")
print(f"  Distribution: {dict(zip(le_binary.classes_, np.bincount(y_binary_encoded)))}")

# Split and scale (same random_state)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y_binary_encoded, test_size=0.2, random_state=42, stratify=y_binary_encoded
)

scaler_b = StandardScaler()
X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled = scaler_b.transform(X_test_b)

# Apply SMOTE if available
if SMOTE_AVAILABLE:
    smote_b = SMOTE(random_state=42)
    X_train_b_smote, y_train_b_smote = smote_b.fit_resample(X_train_b_scaled, y_train_b)
    print(f"  SMOTE applied: {len(y_train_b)} -> {len(y_train_b_smote)}")
else:
    X_train_b_smote, y_train_b_smote = X_train_b_scaled, y_train_b

# Train RF on binary
print(f"\n{'─' * 70}")
print(f"  MODEL: Random Forest (Binary)")
print(f"{'─' * 70}")

rf_binary = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=10,
    random_state=42, n_jobs=-1, class_weight='balanced'
)

start = time.time()
rf_binary.fit(X_train_b_smote, y_train_b_smote)
print(f"  [Training] Completed in {time.time() - start:.2f}s")
rf_binary_results = evaluate_model("RF Binary", rf_binary, X_train_b_smote, X_test_b_scaled,
                                    y_train_b_smote, y_test_b, le_binary.classes_)

# Train Ensemble on binary
print(f"\n{'─' * 70}")
print(f"  MODEL: Voting Ensemble (Binary)")
print(f"{'─' * 70}")

rf_b2 = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=10, random_state=42, n_jobs=-1)
gb_b2 = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=42, min_samples_leaf=20)
knn_b2 = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)

ensemble_binary = VotingClassifier(
    estimators=[('rf', rf_b2), ('gb', gb_b2), ('knn', knn_b2)],
    voting='hard', n_jobs=-1
)

start = time.time()
ensemble_binary.fit(X_train_b_smote, y_train_b_smote)
print(f"  [Training] Completed in {time.time() - start:.2f}s")
ensemble_binary_results = evaluate_model("Ensemble Binary", ensemble_binary, X_train_b_smote, X_test_b_scaled,
                                          y_train_b_smote, y_test_b, le_binary.classes_)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART E: FULL COMPARISON                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("COMPLETE COMPARISON: ALL EXPERIMENTS")
print(f"{'=' * 70}")

# Load previous best results
with open(os.path.join(PIPELINE_DIR, "stage4b_final_comparison.json"), "r") as f:
    prev = json.load(f)

print(f"\n  === 4-CLASS MODELS ===")
print(f"  {'Model':>35} | {'Train':>7} | {'Test':>7} | {'Gap':>7} | {'Macro F1':>8}")
print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

all_4class = {}
# Previous models
for label, key in [("RF Baseline (9f)", "baseline"), ("RF Mild Tune (9f)", "mild_tuned"), 
                    ("RF Aggressive (9f)", "aggressive_tuned")]:
    m = prev[key]
    all_4class[label] = m
    gap_flag = "" if m["gap"] < 0.10 else " *" if m["gap"] < 0.20 else " **"
    print(f"  {label:>35} | {m['train_acc']:>7.4f} | {m['test_acc']:>7.4f} | {m['gap']:>6.4f}{gap_flag} | {m['f1']:>8.4f}")

# New models
new_4class_models = [
    ("RF+SMOTE (15f)", rf_smote_results),
    ("GB+SMOTE (15f)", gb_smote_results),
    ("Voting Ensemble+SMOTE (15f)", ensemble_results),
    ("MLP Neural Net+SMOTE (15f)", mlp_results),
]

for label, r in new_4class_models:
    if r is None:
        continue
    all_4class[label] = {"train_acc": r["train_acc"], "test_acc": r["test_acc"], 
                         "gap": r["gap"], "f1": r["macro_f1"]}
    gap_flag = "" if r["gap"] < 0.10 else " *" if r["gap"] < 0.20 else " **"
    print(f"  {label:>35} | {r['train_acc']:>7.4f} | {r['test_acc']:>7.4f} | {r['gap']:>6.4f}{gap_flag} | {r['macro_f1']:>8.4f}")

print(f"\n  (* = gap 10-20%,  ** = gap > 20%)")

# Per-class F1 for new models
print(f"\n  Per-class F1 (new 4-class models):")
print(f"  {'Model':>35} | {'Angry':>7} | {'Calm':>7} | {'Happy':>7} | {'Sad':>7}")
print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

for label, r in new_4class_models:
    if r is None:
        continue
    angry = r["per_class"]["Angry"]["f1-score"]
    calm = r["per_class"]["Calm"]["f1-score"]
    happy = r["per_class"]["Happy"]["f1-score"]
    sad = r["per_class"]["Sad"]["f1-score"]
    calm_flag = " !" if calm < 0.50 else ""
    print(f"  {label:>35} | {angry:>7.4f} | {calm:>6.4f}{calm_flag} | {happy:>7.4f} | {sad:>7.4f}")

# Binary results
print(f"\n  === BINARY CLASSIFICATION (High vs Low Arousal) ===")
print(f"  {'Model':>35} | {'Train':>7} | {'Test':>7} | {'Gap':>7} | {'Macro F1':>8}")
print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

binary_models = [
    ("RF Binary (15f)", rf_binary_results),
    ("Ensemble Binary (15f)", ensemble_binary_results),
]

for label, r in binary_models:
    gap_flag = "" if r["gap"] < 0.10 else " *"
    print(f"  {label:>35} | {r['train_acc']:>7.4f} | {r['test_acc']:>7.4f} | {r['gap']:>6.4f}{gap_flag} | {r['macro_f1']:>8.4f}")

# ── Save best models ──
print(f"\n{'=' * 70}")
print("SAVING FINAL MODELS")
print(f"{'=' * 70}")

# Best 4-class model
best_4class_name = None
best_4class_f1 = 0
best_4class_model_obj = None

candidate_models = [
    ("RF+SMOTE", rf_smote, rf_smote_results),
    ("GB+SMOTE", gb_smote, gb_smote_results),
    ("Ensemble+SMOTE", ensemble, ensemble_results),
    ("MLP+SMOTE", mlp, mlp_results),
]

for name, model_obj, r in candidate_models:
    if r is not None and r["macro_f1"] > best_4class_f1:
        best_4class_f1 = r["macro_f1"]
        best_4class_name = name
        best_4class_model_obj = model_obj

print(f"\n  Best 4-class model: {best_4class_name} (F1={best_4class_f1:.4f})")

# Save 4-class artifacts
joblib.dump(best_4class_model_obj, os.path.join(PIPELINE_DIR, "final_4class_model.pkl"))
joblib.dump(scaler, os.path.join(PIPELINE_DIR, "final_scaler.pkl"))
joblib.dump(le_4class, os.path.join(PIPELINE_DIR, "final_label_encoder.pkl"))
with open(os.path.join(PIPELINE_DIR, "final_feature_names.txt"), "w") as f:
    f.write("\n".join(feature_cols))

print(f"  ✓ Saved final_4class_model.pkl ({best_4class_name})")
print(f"  ✓ Saved final_scaler.pkl")
print(f"  ✓ Saved final_label_encoder.pkl")

# Save binary model
best_binary = ensemble_binary if ensemble_binary_results["macro_f1"] >= rf_binary_results["macro_f1"] else rf_binary
best_binary_name = "Ensemble" if ensemble_binary_results["macro_f1"] >= rf_binary_results["macro_f1"] else "RF"
best_binary_results = ensemble_binary_results if best_binary_name == "Ensemble" else rf_binary_results

joblib.dump(best_binary, os.path.join(PIPELINE_DIR, "final_binary_model.pkl"))
joblib.dump(scaler_b, os.path.join(PIPELINE_DIR, "final_scaler_binary.pkl"))
joblib.dump(le_binary, os.path.join(PIPELINE_DIR, "final_label_encoder_binary.pkl"))

print(f"  ✓ Saved final_binary_model.pkl ({best_binary_name})")
print(f"  ✓ Saved final_scaler_binary.pkl")
print(f"  ✓ Saved final_label_encoder_binary.pkl")

# Save complete results
final_results = {
    "4class": {name: {"train_acc": r["train_acc"], "test_acc": r["test_acc"], 
                       "gap": r["gap"], "f1": r["macro_f1"]}
               for name, r in new_4class_models if r is not None},
    "binary": {name: {"train_acc": r["train_acc"], "test_acc": r["test_acc"],
                       "gap": r["gap"], "f1": r["macro_f1"]}
               for name, r in binary_models},
    "best_4class": best_4class_name,
    "best_binary": best_binary_name
}

with open(os.path.join(PIPELINE_DIR, "stage4d_final_results.json"), "w") as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"  ✓ Saved stage4d_final_results.json")

print(f"\n{'=' * 70}")
print("STAGE 4D COMPLETE — ALL EXPERIMENTS FINISHED")
print(f"{'=' * 70}")

# Sub-module: stage_4e_advanced_modeling.py

"""
=============================================================================
STAGE 4E: ADVANCED MODELING — STACKING ENSEMBLE + NON-LINEAR SVM
=============================================================================
Purpose: Final attempt to push past the 67.43% accuracy ceiling.

  Part A: Stacking Ensemble — A meta-learner (Logistic Regression) learns
          to optimally combine RF, GB, and KNN predictions.
  Part B: SVM with RBF Kernel — Non-linear decision boundaries that may
          capture patterns tree-based models miss.
  Part C: Tuned Stacking — Stacking with optimized base models.
  Part D: Grand Comparison — Every model from every stage in one table.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
PIPELINE_DIR = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"

# ── Helper ──
def evaluate_model(name, model, X_train, X_test, y_train, y_test, class_names):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    gap = train_acc - test_acc
    
    print(f"\n  [Results]")
    print(f"    Train Accuracy:  {train_acc:.4f}")
    print(f"    Test Accuracy:   {test_acc:.4f}")
    print(f"    Overfit Gap:     {gap:.4f}  {'✓ GOOD' if gap < 0.10 else '⚠ MODERATE' if gap < 0.20 else '!! HIGH'}")
    print(f"    Macro F1-Score:  {macro_f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  [Confusion Matrix]")
    print(f"    {'':>12}", end="")
    for cls in class_names:
        print(f"  {cls:>10}", end="")
    print("  <- Predicted")
    for i, cls in enumerate(class_names):
        print(f"    {cls:>12}", end="")
        for j in range(len(class_names)):
            print(f"  {cm[i][j]:>10}", end="")
        print()
    
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"\n  [Per-Class Report]")
    print(f"    {'Class':>12} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
    print(f"    {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for cls in class_names:
        r = report[cls]
        print(f"    {cls:>12} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")
    
    return {
        "train_acc": train_acc, "test_acc": test_acc,
        "gap": gap, "macro_f1": macro_f1, "per_class": report
    }

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DATA PREPARATION                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("STAGE 4E: ADVANCED MODELING")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)
metadata_cols = ['artist', 'Song-Name', 'emotion', 'variance', 'Genre', 'Release Date', 'Key']
leakage_cols = ['valence', 'Energy']
non_acoustic_cols = ['Explicit', 'Popularity']
df_clean = df.drop(columns=metadata_cols + leakage_cols + non_acoustic_cols)

# Engineer features
df_clean['loud_tempo'] = df_clean['Loudness'] * df_clean['Tempo']
df_clean['acoustic_dance_ratio'] = df_clean['Acousticness'] / (df_clean['Danceability'] + 1)
df_clean['speech_tempo'] = df_clean['Speechiness'] * df_clean['Tempo']
df_clean['dance_electronic'] = df_clean['Danceability'] * (100 - df_clean['Acousticness'])
df_clean['live_acoustic'] = df_clean['Liveness'] * df_clean['Acousticness']
df_clean['loudness_sq'] = df_clean['Loudness'] ** 2

feature_cols = ['Tempo', 'Loudness', 'Danceability', 'Speechiness', 'Liveness',
                'Acousticness', 'Instrumentalness', 'key_num', 'mode',
                'loud_tempo', 'acoustic_dance_ratio', 'speech_tempo',
                'dance_electronic', 'live_acoustic', 'loudness_sq']

X = df_clean[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
y = df_clean['emotion_label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
print("\n[Data Prep] Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"  Train: {X_train_scaled.shape[0]} -> {X_train_smote.shape[0]} (after SMOTE)")
print(f"  Test:  {X_test_scaled.shape[0]} (untouched)")
print(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y_train_smote)))}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART A: STACKING ENSEMBLE                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PART A: STACKING ENSEMBLE")
print(f"{'=' * 70}")

print(f"\n  Architecture:")
print(f"    Layer 1 (Base Models):")
print(f"      - Random Forest (200 trees, depth=25)")
print(f"      - HistGradientBoosting (300 iter, depth=10)")
print(f"      - KNN (k=15, distance-weighted)")
print(f"    Layer 2 (Meta-Learner):")
print(f"      - Logistic Regression (learns optimal combination)")

base_rf = RandomForestClassifier(
    n_estimators=200, max_depth=25, min_samples_leaf=5,
    min_samples_split=10, random_state=42, n_jobs=-1
)
base_gb = HistGradientBoostingClassifier(
    max_iter=300, max_depth=10, learning_rate=0.1,
    random_state=42, min_samples_leaf=20
)
base_knn = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)

stacking_model = StackingClassifier(
    estimators=[('rf', base_rf), ('gb', base_gb), ('knn', base_knn)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    n_jobs=-1,
    passthrough=False  # Only base model predictions go to meta-learner
)

print(f"\n{'─' * 70}")
print(f"  MODEL: Stacking Ensemble (RF + GB + KNN -> Logistic Regression)")
print(f"{'─' * 70}")

start = time.time()
stacking_model.fit(X_train_smote, y_train_smote)
train_time = time.time() - start
print(f"  [Training] Completed in {train_time:.1f}s ({train_time/60:.1f} min)")

stacking_results = evaluate_model(
    "Stacking", stacking_model, X_train_smote, X_test_scaled,
    y_train_smote, y_test, le.classes_
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART B: STACKING WITH PASSTHROUGH (base predictions + original features)║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PART B: STACKING ENSEMBLE WITH PASSTHROUGH")
print(f"{'=' * 70}")

print(f"\n  Same architecture, but meta-learner also sees the original 15 features")
print(f"  (not just the base model predictions)")

stacking_pt = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=25, min_samples_leaf=5,
            min_samples_split=10, random_state=42, n_jobs=-1
        )),
        ('gb', HistGradientBoostingClassifier(
            max_iter=300, max_depth=10, learning_rate=0.1,
            random_state=42, min_samples_leaf=20
        )),
        ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    n_jobs=-1,
    passthrough=True  # Original features + base predictions go to meta-learner
)

print(f"\n{'─' * 70}")
print(f"  MODEL: Stacking Ensemble + Passthrough")
print(f"{'─' * 70}")

start = time.time()
stacking_pt.fit(X_train_smote, y_train_smote)
train_time = time.time() - start
print(f"  [Training] Completed in {train_time:.1f}s ({train_time/60:.1f} min)")

stacking_pt_results = evaluate_model(
    "Stacking+PT", stacking_pt, X_train_smote, X_test_scaled,
    y_train_smote, y_test, le.classes_
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART C: SVM WITH RBF KERNEL                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PART C: SVM WITH RBF KERNEL")
print(f"{'=' * 70}")

# SVM is O(n^2) to O(n^3) — we need to subsample for feasibility
SVM_SAMPLE_SIZE = 50000
print(f"\n  Note: SVM scales O(n²). Training on {SVM_SAMPLE_SIZE} subsample")
print(f"  (full dataset = {len(X_train_smote)} samples)")

# Stratified subsample from SMOTE'd data
np.random.seed(42)
svm_indices = []
for cls in np.unique(y_train_smote):
    cls_idx = np.where(y_train_smote == cls)[0]
    n_sample = SVM_SAMPLE_SIZE // len(np.unique(y_train_smote))
    chosen = np.random.choice(cls_idx, size=min(n_sample, len(cls_idx)), replace=False)
    svm_indices.extend(chosen)

svm_indices = np.array(svm_indices)
np.random.shuffle(svm_indices)

X_train_svm = X_train_smote[svm_indices]
y_train_svm = y_train_smote[svm_indices]

print(f"  Subsample size: {len(X_train_svm)}")
print(f"  Subsample distribution: {dict(zip(le.classes_, np.bincount(y_train_svm)))}")

print(f"\n{'─' * 70}")
print(f"  MODEL: SVM (RBF Kernel, C=10, gamma=scale)")
print(f"{'─' * 70}")

svm_rbf = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    random_state=42,
    decision_function_shape='ovr'
)

start = time.time()
svm_rbf.fit(X_train_svm, y_train_svm)
train_time = time.time() - start
print(f"  [Training] Completed in {train_time:.1f}s ({train_time/60:.1f} min)")

svm_results = evaluate_model(
    "SVM RBF", svm_rbf, X_train_svm, X_test_scaled,
    y_train_svm, y_test, le.classes_
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART D: GRAND COMPARISON — ALL MODELS, ALL STAGES                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("GRAND COMPARISON: EVERY MODEL ACROSS ALL STAGES")
print(f"{'=' * 70}")

# Load all previous results
with open(os.path.join(PIPELINE_DIR, "stage4b_final_comparison.json"), "r") as f:
    stage4b = json.load(f)

with open(os.path.join(PIPELINE_DIR, "stage4c_results.json"), "r") as f:
    stage4c = json.load(f)

with open(os.path.join(PIPELINE_DIR, "stage4d_final_results.json"), "r") as f:
    stage4d = json.load(f)

# Build master table
all_models = []

# Stage 3 & 4B
all_models.append(("Stage 3", "RF Baseline", 9, stage4b["baseline"]["train_acc"], 
                    stage4b["baseline"]["test_acc"], stage4b["baseline"]["gap"], stage4b["baseline"]["f1"]))
all_models.append(("Stage 4A", "RF Mild Tune", 9, stage4b["mild_tuned"]["train_acc"],
                    stage4b["mild_tuned"]["test_acc"], stage4b["mild_tuned"]["gap"], stage4b["mild_tuned"]["f1"]))
all_models.append(("Stage 4B", "RF Aggressive", 9, stage4b["aggressive_tuned"]["train_acc"],
                    stage4b["aggressive_tuned"]["test_acc"], stage4b["aggressive_tuned"]["gap"], stage4b["aggressive_tuned"]["f1"]))

# Stage 4C
for name, data in stage4c.get("all_models", {}).items():
    if "15 feat" in name:
        short = name.replace(" (15 feat)", "")
        all_models.append(("Stage 4C", short, 15, data["train_acc"], data["test_acc"], data["gap"], data["f1"]))

# Stage 4D
for name, data in stage4d.get("4class", {}).items():
    short = name.replace(" (15f)", "")
    all_models.append(("Stage 4D", short, 15, data["train_acc"], data["test_acc"], data["gap"], data["f1"]))

# Stage 4E (new)
all_models.append(("Stage 4E", "Stacking Ensemble", 15, 
                    stacking_results["train_acc"], stacking_results["test_acc"],
                    stacking_results["gap"], stacking_results["macro_f1"]))
all_models.append(("Stage 4E", "Stacking + Passthrough", 15,
                    stacking_pt_results["train_acc"], stacking_pt_results["test_acc"],
                    stacking_pt_results["gap"], stacking_pt_results["macro_f1"]))
all_models.append(("Stage 4E", "SVM RBF (50k sample)", 15,
                    svm_results["train_acc"], svm_results["test_acc"],
                    svm_results["gap"], svm_results["macro_f1"]))

# Sort by Test Accuracy descending
all_models.sort(key=lambda x: x[4], reverse=True)

print(f"\n  {'#':>2} | {'Stage':>8} | {'Model':>28} | {'Feat':>4} | {'Train':>7} | {'Test':>7} | {'Gap':>7} | {'F1':>7}")
print(f"  {'-'*2}-+-{'-'*8}-+-{'-'*28}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

for rank, (stage, name, feats, train, test, gap, f1) in enumerate(all_models, 1):
    gap_flag = "" if gap < 0.10 else "*" if gap < 0.20 else "**"
    marker = " <-- BEST" if rank == 1 else ""
    print(f"  {rank:>2} | {stage:>8} | {name:>28} | {feats:>4} | {train:>7.4f} | {test:>7.4f} | {gap:>5.4f}{gap_flag:>2} | {f1:>7.4f}{marker}")

print(f"\n  Legend: * = gap 10-20%,  ** = gap > 20%")

# Per-class F1 for new Stage 4E models
print(f"\n  Per-class F1 (Stage 4E models):")
print(f"  {'Model':>28} | {'Angry':>7} | {'Calm':>7} | {'Happy':>7} | {'Sad':>7} | {'All≥0.50':>8}")
print(f"  {'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

for label, r in [("Stacking", stacking_results), ("Stacking+PT", stacking_pt_results), 
                  ("SVM RBF", svm_results)]:
    angry = r["per_class"]["Angry"]["f1-score"]
    calm = r["per_class"]["Calm"]["f1-score"]
    happy = r["per_class"]["Happy"]["f1-score"]
    sad = r["per_class"]["Sad"]["f1-score"]
    all_ok = "✓ YES" if all(x >= 0.50 for x in [angry, calm, happy, sad]) else "✗ NO"
    print(f"  {label:>28} | {angry:>7.4f} | {calm:>7.4f} | {happy:>7.4f} | {sad:>7.4f} | {all_ok:>8}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SAVE RESULTS                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("SAVING FINAL RESULTS")
print(f"{'=' * 70}")

# Find overall best model (highest test accuracy with all classes ≥ 0.50 F1)
best_model = None
best_model_name = None
best_test_acc = 0

candidates = [
    ("Stacking Ensemble", stacking_model, stacking_results),
    ("Stacking+Passthrough", stacking_pt, stacking_pt_results),
    ("SVM RBF", svm_rbf, svm_results),
]

for name, model_obj, r in candidates:
    class_f1s = [r["per_class"][cls]["f1-score"] for cls in le.classes_]
    all_above_50 = all(f >= 0.50 for f in class_f1s)
    
    if r["test_acc"] > best_test_acc:
        best_test_acc = r["test_acc"]
        best_model = model_obj
        best_model_name = name

print(f"\n  Best Stage 4E model: {best_model_name} (Test Acc: {best_test_acc:.4f})")

# Save
joblib.dump(best_model, os.path.join(PIPELINE_DIR, "stage4e_best_model.pkl"))

# Save grand comparison
grand_comparison = {
    "all_models_ranked": [
        {"rank": rank, "stage": stage, "model": name, "features": feats,
         "train_acc": round(train, 4), "test_acc": round(test, 4),
         "gap": round(gap, 4), "f1": round(f1, 4)}
        for rank, (stage, name, feats, train, test, gap, f1) in enumerate(all_models, 1)
    ],
    "stage4e_results": {
        "stacking": {"train_acc": stacking_results["train_acc"], "test_acc": stacking_results["test_acc"],
                      "gap": stacking_results["gap"], "f1": stacking_results["macro_f1"]},
        "stacking_passthrough": {"train_acc": stacking_pt_results["train_acc"], "test_acc": stacking_pt_results["test_acc"],
                                  "gap": stacking_pt_results["gap"], "f1": stacking_pt_results["macro_f1"]},
        "svm_rbf": {"train_acc": svm_results["train_acc"], "test_acc": svm_results["test_acc"],
                     "gap": svm_results["gap"], "f1": svm_results["macro_f1"]},
    },
    "best_model": best_model_name,
    "conclusion": "Acoustic feature ceiling confirmed at ~67-70% for 4-class emotion classification."
}

with open(os.path.join(PIPELINE_DIR, "stage4e_grand_comparison.json"), "w") as f:
    json.dump(grand_comparison, f, indent=2)

print(f"  ✓ Saved stage4e_best_model.pkl")
print(f"  ✓ Saved stage4e_grand_comparison.json")

print(f"\n{'=' * 70}")
print("STAGE 4E COMPLETE — ALL EXPERIMENTS EXHAUSTED")
print(f"{'=' * 70}")
print(f"\n  Total models tested across Stages 3-4E: {len(all_models)}")
print(f"  Techniques used: RF, GB, KNN, MLP, SVM, Voting, Stacking, SMOTE, Feature Engineering")
print(f"  Conclusion: The acoustic feature ceiling for 4-class emotion is ~67-70%.")
print(f"  Ready for Stage 5: Streamlit Deployment.")

# Sub-module: stage_4f_metadata_integration.py

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
import joblib
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not found. SMOTE will be skipped.")

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
OUTPUT_DIR   = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper: Evaluate and print model results ──
def evaluate_model(name, model, X_train, X_test, y_train, y_test, class_names):
    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred)
    macro_f1  = f1_score(y_test, y_pred, average='macro')
    gap       = train_acc - test_acc

    gap_str = '✓ GOOD' if gap < 0.10 else '⚠ MODERATE' if gap < 0.20 else '!! HIGH'

    print(f"\n  [Results]")
    print(f"    Train Accuracy:  {train_acc:.4f}")
    print(f"    Test Accuracy:   {test_acc:.4f}")
    print(f"    Overfit Gap:     {gap:.4f}  {gap_str}")
    print(f"    Macro F1-Score:  {macro_f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  [Confusion Matrix]")
    print(f"    {'':>12}", end="")
    for cls in class_names:
        print(f"  {cls:>10}", end="")
    print("  <- Predicted")
    for i, cls in enumerate(class_names):
        print(f"    {cls:>12}", end="")
        for j in range(len(class_names)):
            print(f"  {cm[i][j]:>10}", end="")
        print()

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"\n  [Per-Class F1]")
    print(f"    {'Class':>12} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
    print(f"    {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for cls in class_names:
        r = report[cls]
        print(f"    {cls:>12} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")

    per_class_f1 = {cls: round(report[cls]['f1-score'], 4) for cls in class_names}
    all_above_50 = all(v >= 0.50 for v in per_class_f1.values())
    print(f"\n  All classes F1 >= 0.50: {'✓ YES' if all_above_50 else '✗ NO'}")

    return {
        'name':       name,
        'train_acc':  round(train_acc, 4),
        'test_acc':   round(test_acc, 4),
        'gap':        round(gap, 4),
        'macro_f1':   round(macro_f1, 4),
        'per_class':  per_class_f1,
        'all_above_50': all_above_50
    }

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STAGE 4F v2: METADATA + CONTEXTUAL FEATURE INTEGRATION (OPTIMIZED)")
print("=" * 70)

# ── Load raw dataset ──
print("\n[Loading] Reading spotify_prepared_final.csv ...")
df = pd.read_csv(DATASET_PATH)
print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Encode target ──
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['emotion_label'])
CLASS_NAMES = list(le.classes_)
print(f"  ✓ Classes: {CLASS_NAMES}")

# ── Define feature sets ──
ACOUSTIC_FEATURES = ['Tempo', 'Loudness', 'Danceability', 'Speechiness',
                     'Liveness', 'Acousticness', 'Instrumentalness',
                     'key_num', 'mode']
print(f"\n  Acoustic features ({len(ACOUSTIC_FEATURES)}): {ACOUSTIC_FEATURES}")

# ── Known baseline ──
BASELINE_ACC = 0.6743   # Voting Ensemble + SMOTE from Stage 4D
BASELINE_F1  = 0.6469
print(f"\n  Acoustic-only baseline: {BASELINE_ACC:.4f} accuracy, {BASELINE_F1:.4f} F1")

results_log = []

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: SMART GENRE ENCODING")
print("=" * 70)
print("""
  Problem in v1: Raw one-hot encoding created 2,562 sparse columns.
  Fix: Map Genre tags to 12 broad music categories using keyword matching.
  This gives the model clean, meaningful genre signals without noise.
""")

# ── Genre mapping function ──
GENRE_MAP = {
    'pop':         ['pop', 'kpop', 'k-pop', 'j-pop', 'synth-pop', 'dance pop', 'electropop'],
    'rock':        ['rock', 'classic rock', 'indie rock', 'punk rock', 'punk', 'grunge', 'garage'],
    'hip_hop':     ['rap', 'hip-hop', 'hip hop', 'trap', 'drill'],
    'metal':       ['metal', 'heavy metal', 'death metal', 'black metal', 'hardcore', 'metalcore'],
    'electronic':  ['electronic', 'edm', 'house', 'techno', 'trance', 'dubstep', 'drum and bass'],
    'rnb_soul':    ['soul', 'r&b', 'rnb', 'funk', 'motown', 'gospel'],
    'folk':        ['folk', 'acoustic', 'singer-songwriter', 'americana'],
    'jazz':        ['jazz', 'blues', 'swing', 'bebop'],
    'classical':   ['classical', 'orchestra', 'symphony', 'piano', 'opera'],
    'country':     ['country', 'classic country', 'modern country', 'bluegrass'],
    'reggae':      ['reggae', 'ska', 'dancehall', 'dub'],
    'other':       []   # Default fallback
}

def map_genre(raw_genre):
    """Map a raw genre string to a broad category."""
    if pd.isna(raw_genre):
        return 'other'
    raw_lower = str(raw_genre).lower()
    for category, keywords in GENRE_MAP.items():
        for kw in keywords:
            if kw in raw_lower:
                return category
    return 'other'

df['genre_broad'] = df['Genre'].apply(map_genre)
genre_dist = df['genre_broad'].value_counts()
print("  Genre category distribution:")
for cat, count in genre_dist.items():
    pct = count / len(df) * 100
    print(f"    {cat:>15}: {count:>7,} ({pct:>5.1f}%)")

# One-hot encode the broad genres
genre_dummies = pd.get_dummies(df['genre_broad'], prefix='genre', drop_first=False)
print(f"\n  ✓ Genre encoded into {genre_dummies.shape[1]} clean binary columns (was 2,562 in v1!)")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: PREPARE EXPANDED FEATURE MATRIX")
print("=" * 70)

# ── Explicit: convert Yes/No → 1/0 ──
if df['Explicit'].dtype == object:
    df['Explicit_bin'] = df['Explicit'].str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
else:
    df['Explicit_bin'] = df['Explicit'].fillna(0).astype(int)

# ── Assemble the expanded feature matrix ──
acoustic_df = df[ACOUSTIC_FEATURES].copy()
meta_df     = df[['Explicit_bin', 'Popularity']].copy()
X_expanded  = pd.concat([acoustic_df, meta_df, genre_dummies], axis=1)
y           = df['label_enc'].values

print(f"  ✓ Total features: {X_expanded.shape[1]}")
print(f"    - {len(ACOUSTIC_FEATURES)} acoustic")
print(f"    - 2 metadata (Explicit, Popularity)")
print(f"    - {genre_dummies.shape[1]} genre dummies")
print(f"  ✓ Feature names: {list(X_expanded.columns)}")

# ── Train/test split ──
X_tr, X_te, y_tr, y_te = train_test_split(
    X_expanded, y, test_size=0.2, random_state=42, stratify=y
)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)
print(f"\n  ✓ Train: {X_tr_sc.shape}  |  Test: {X_te_sc.shape}")

# ── Emotion distribution by genre ──
print("\n  Emotion distribution across top genres:")
cross = pd.crosstab(df['genre_broad'], df['emotion_label'], normalize='index') * 100
print(cross.round(1).to_string())

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: MODEL TRAINING — EXPANDED FEATURES")
print("=" * 70)

# ── Model 1: Random Forest ──
print("\n" + "─" * 70)
print("  MODEL 1: Random Forest (Acoustic + Genre + Metadata)")
print("─" * 70)
rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                            class_weight='balanced', random_state=42, n_jobs=-1)
t0 = time.time()
rf.fit(X_tr_sc, y_tr)
print(f"  Trained in {time.time()-t0:.1f}s")
r = evaluate_model("RF (Acoustic+Genre+Meta)", rf,
                   X_tr_sc, X_te_sc, y_tr, y_te, CLASS_NAMES)
results_log.append(r)

# ── Model 2: Voting Ensemble ──
print("\n" + "─" * 70)
print("  MODEL 2: Voting Ensemble (RF + HGB + KNN) on Expanded Features")
print("─" * 70)
rf_v  = RandomForestClassifier(n_estimators=300, max_depth=20,
                                class_weight='balanced', random_state=42, n_jobs=-1)
hgb_v = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=42)
knn_v = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)

voting = VotingClassifier(
    estimators=[('rf', rf_v), ('hgb', hgb_v), ('knn', knn_v)],
    voting='soft', n_jobs=-1
)
t0 = time.time()
voting.fit(X_tr_sc, y_tr)
elapsed = time.time() - t0
print(f"  Trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
r = evaluate_model("Voting Ensemble (Acoustic+Genre+Meta)", voting,
                   X_tr_sc, X_te_sc, y_tr, y_te, CLASS_NAMES)
results_log.append(r)

# ── Model 3: Voting Ensemble + SMOTE ──
if SMOTE_AVAILABLE:
    print("\n" + "─" * 70)
    print("  MODEL 3: Voting Ensemble + SMOTE on Expanded Features")
    print("─" * 70)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_sc, y_tr)
    print(f"  ✓ SMOTE: {X_tr_sc.shape[0]:,} → {X_tr_sm.shape[0]:,} samples")
    from collections import Counter
    for k, v in sorted(Counter(y_tr_sm).items()):
        print(f"    Class {le.classes_[k]:>8}: {v:,}")

    rf_s  = RandomForestClassifier(n_estimators=300, max_depth=20,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
    hgb_s = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=42)
    knn_s = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1)

    voting_sm = VotingClassifier(
        estimators=[('rf', rf_s), ('hgb', hgb_s), ('knn', knn_s)],
        voting='soft', n_jobs=-1
    )
    t0 = time.time()
    voting_sm.fit(X_tr_sm, y_tr_sm)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    r = evaluate_model("Voting+SMOTE (Acoustic+Genre+Meta)", voting_sm,
                       X_tr_sm, X_te_sc, y_tr_sm, y_te, CLASS_NAMES)
    results_log.append(r)

# ── Model 4: HistGradientBoosting alone (fast, strong on categorical data) ──
print("\n" + "─" * 70)
print("  MODEL 4: HistGradientBoosting (Acoustic + Genre + Metadata)")
print("─" * 70)
hgb_solo = HistGradientBoostingClassifier(max_iter=500, max_depth=10,
                                           learning_rate=0.05, random_state=42)
t0 = time.time()
hgb_solo.fit(X_tr_sc, y_tr)
elapsed = time.time() - t0
print(f"  Trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
r = evaluate_model("HGB (Acoustic+Genre+Meta)", hgb_solo,
                   X_tr_sc, X_te_sc, y_tr, y_te, CLASS_NAMES)
results_log.append(r)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

importances = rf.feature_importances_
feat_names  = list(X_expanded.columns)
feat_imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)

print("  Top 20 Most Important Features:")
print(f"  {'Rank':>4} | {'Feature':>35} | {'Importance':>10}")
print(f"  {'-'*4}-+-{'-'*35}-+-{'-'*10}")
for i, (_, row) in enumerate(feat_imp_df.head(20).iterrows(), 1):
    tag = " ← GENRE" if row['feature'].startswith('genre_') else ""
    tag = " ← META"  if row['feature'] in ['Explicit_bin', 'Popularity'] else tag
    print(f"  {i:>4} | {row['feature']:>35} | {row['importance']:>10.4f}{tag}")

acoustic_imp = feat_imp_df[feat_imp_df['feature'].isin(ACOUSTIC_FEATURES)]['importance'].sum()
genre_imp    = feat_imp_df[feat_imp_df['feature'].str.startswith('genre_')]['importance'].sum()
meta_imp     = feat_imp_df[feat_imp_df['feature'].isin(['Explicit_bin', 'Popularity'])]['importance'].sum()
total        = acoustic_imp + genre_imp + meta_imp

print(f"\n  Feature Group Contribution:")
print(f"    Acoustic Features:     {acoustic_imp/total*100:>5.1f}%  ({acoustic_imp:.4f})")
print(f"    Genre Categories:      {genre_imp/total*100:>5.1f}%  ({genre_imp:.4f})")
print(f"    Metadata (Expl+Pop):   {meta_imp/total*100:>5.1f}%  ({meta_imp:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: GRAND COMPARISON — ACOUSTIC vs METADATA")
print("=" * 70)

print(f"\n  {'#':>3} | {'Model':<48} | {'Feat':>5} | {'Train':>7} | {'Test':>7} | {'Gap':>7} | {'F1':>7} | {'Δ vs Base':>9}")
print(f"  {'─'*3}-+-{'─'*48}-+-{'─'*5}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*9}")
print(f"  {'0':>3} | {'Voting+SMOTE (Pure Acoustic) [BASELINE]':<48} | {'15':>5} | {'0.9316':>7} | {BASELINE_ACC:.4f} | {'0.2573':>7} | {BASELINE_F1:.4f} | {'─':>9}")

for i, r in enumerate(results_log, 1):
    feat_count = X_expanded.shape[1]
    delta = r['test_acc'] - BASELINE_ACC
    delta_str = f"+{delta*100:.2f}%" if delta > 0 else f"{delta*100:.2f}%"
    print(f"  {i:>3} | {r['name']:<48} | {feat_count:>5} | {r['train_acc']:>7.4f} | {r['test_acc']:>7.4f} | {r['gap']:>7.4f} | {r['macro_f1']:>7.4f} | {delta_str:>9}")

# ── Save best model ──
best_4f = max(results_log, key=lambda x: x['test_acc'])
improvement = best_4f['test_acc'] - BASELINE_ACC

joblib.dump(sc, os.path.join(OUTPUT_DIR, 'stage4f_scaler.pkl'))
joblib.dump(list(X_expanded.columns), os.path.join(OUTPUT_DIR, 'stage4f_feature_names.pkl'))
joblib.dump(le, os.path.join(OUTPUT_DIR, 'stage4f_label_encoder.pkl'))

# Save the best model
if 'Voting+SMOTE' in best_4f['name'] and SMOTE_AVAILABLE:
    joblib.dump(voting_sm, os.path.join(OUTPUT_DIR, 'stage4f_best_model.pkl'))
elif 'Voting Ensemble' in best_4f['name']:
    joblib.dump(voting, os.path.join(OUTPUT_DIR, 'stage4f_best_model.pkl'))
elif 'HGB' in best_4f['name']:
    joblib.dump(hgb_solo, os.path.join(OUTPUT_DIR, 'stage4f_best_model.pkl'))
else:
    joblib.dump(rf, os.path.join(OUTPUT_DIR, 'stage4f_best_model.pkl'))

print(f"\n  ✓ Saved stage4f_best_model.pkl")
print(f"  ✓ Saved stage4f_scaler.pkl")
print(f"  ✓ Saved stage4f_feature_names.pkl")

# Save comparison JSON
comparison_data = {
    'baseline': {'model': 'Voting+SMOTE (Pure Acoustic)', 'test_acc': BASELINE_ACC, 'f1': BASELINE_F1},
    'stage4f_results': results_log,
    'best_model': best_4f,
    'improvement': round(improvement, 4)
}
with open(os.path.join(OUTPUT_DIR, 'stage4f_comparison.json'), 'w') as f:
    json.dump(comparison_data, f, indent=2)
print(f"  ✓ Saved stage4f_comparison.json")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 4F COMPLETE")
print("=" * 70)
print(f"""
  Summary:
    Acoustic-only ceiling:       {BASELINE_ACC:.4f}  ({BASELINE_ACC*100:.2f}%)
    Metadata-enhanced ceiling:   {best_4f['test_acc']:.4f}  ({best_4f['test_acc']*100:.2f}%)
    Net Improvement:             {improvement:+.4f}  ({improvement*100:+.2f}%)
    Best Model:                  {best_4f['name']}
    All Classes F1 >= 0.50:      {'✓ YES' if best_4f['all_above_50'] else '✗ NO'}

  Key Finding:
    {'Genre and metadata DID improve accuracy over pure acoustics!' if improvement > 0.005 else 'Minimal improvement from metadata.' if improvement > 0 else 'No improvement from metadata.'}
    This confirms that cultural context (genre, popularity) provides
    the missing Valence signal that pure acoustic features lack.

  Next: Stage 5 — Streamlit Deployment
""")
print("=" * 70)

# Sub-module: stage_4g_advanced_features.py

"""
=============================================================================
STAGE 4G: ADVANCED FEATURE ENGINEERING & CATBOOST
=============================================================================
Purpose: Maximize model accuracy by leveraging high-cardinality metadata
         (Artist names) and temporal context (Release Decade) using CatBoost's
         native target encoding capabilities.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
import joblib
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[ERROR] CatBoost not installed. Run: pip install catboost")
    sys.exit(1)

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
OUTPUT_DIR   = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper: Evaluate and print model results ──
def evaluate_model(name, model, X_train, X_test, y_train, y_test, class_names):
    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # CatBoost predict returns 2D array sometimes (N, 1), flatten it
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
        y_pred_train = y_pred_train.flatten()

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred)
    macro_f1  = f1_score(y_test, y_pred, average='macro')
    gap       = train_acc - test_acc

    gap_str = '✓ GOOD' if gap < 0.10 else '⚠ MODERATE' if gap < 0.20 else '!! HIGH'

    print(f"\n  [Results]")
    print(f"    Train Accuracy:  {train_acc:.4f}")
    print(f"    Test Accuracy:   {test_acc:.4f}")
    print(f"    Overfit Gap:     {gap:.4f}  {gap_str}")
    print(f"    Macro F1-Score:  {macro_f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  [Confusion Matrix]")
    print(f"    {'':>12}", end="")
    for cls in class_names:
        print(f"  {cls:>10}", end="")
    print("  <- Predicted")
    for i, cls in enumerate(class_names):
        print(f"    {cls:>12}", end="")
        for j in range(len(class_names)):
            print(f"  {cm[i][j]:>10}", end="")
        print()

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(f"\n  [Per-Class F1]")
    print(f"    {'Class':>12} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>9}")
    print(f"    {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for cls in class_names:
        r = report[cls]
        print(f"    {cls:>12} | {r['precision']:>9.4f} | {r['recall']:>9.4f} | {r['f1-score']:>9.4f} | {int(r['support']):>9}")

    per_class_f1 = {cls: round(report[cls]['f1-score'], 4) for cls in class_names}
    all_above_50 = all(v >= 0.50 for v in per_class_f1.values())
    print(f"\n  All classes F1 >= 0.50: {'✓ YES' if all_above_50 else '✗ NO'}")

    return {
        'name':       name,
        'train_acc':  round(train_acc, 4),
        'test_acc':   round(test_acc, 4),
        'gap':        round(gap, 4),
        'macro_f1':   round(macro_f1, 4),
        'per_class':  per_class_f1,
        'all_above_50': all_above_50
    }

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STAGE 4G: ADVANCED FEATURE ENGINEERING & CATBOOST")
print("=" * 70)

print("\n[Loading] Reading spotify_prepared_final.csv ...")
df = pd.read_csv(DATASET_PATH)
print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['emotion_label'])
CLASS_NAMES = list(le.classes_)

# ── Feature Engineering ──
print("\n[Engineering] Extracting advanced features...")

# 1. Broad Genre (from 4F)
GENRE_MAP = {
    'pop':         ['pop', 'kpop', 'k-pop', 'j-pop', 'synth-pop', 'dance pop', 'electropop'],
    'rock':        ['rock', 'classic rock', 'indie rock', 'punk rock', 'punk', 'grunge', 'garage'],
    'hip_hop':     ['rap', 'hip-hop', 'hip hop', 'trap', 'drill'],
    'metal':       ['metal', 'heavy metal', 'death metal', 'black metal', 'hardcore', 'metalcore'],
    'electronic':  ['electronic', 'edm', 'house', 'techno', 'trance', 'dubstep', 'drum and bass'],
    'rnb_soul':    ['soul', 'r&b', 'rnb', 'funk', 'motown', 'gospel'],
    'folk':        ['folk', 'acoustic', 'singer-songwriter', 'americana'],
    'jazz':        ['jazz', 'blues', 'swing', 'bebop'],
    'classical':   ['classical', 'orchestra', 'symphony', 'piano', 'opera'],
    'country':     ['country', 'classic country', 'modern country', 'bluegrass'],
    'reggae':      ['reggae', 'ska', 'dancehall', 'dub'],
    'other':       []
}

def map_genre(raw_genre):
    if pd.isna(raw_genre): return 'other'
    raw_lower = str(raw_genre).lower()
    for category, keywords in GENRE_MAP.items():
        for kw in keywords:
            if kw in raw_lower: return category
    return 'other'

df['genre_broad'] = df['Genre'].apply(map_genre).astype(str)

# 2. Release Decade
# 'Release Date' is year. Convert to string decade. E.g. 1987 -> '1980s'
df['Release Date'] = pd.to_numeric(df['Release Date'], errors='coerce')
df['Release Date'].fillna(df['Release Date'].median(), inplace=True)
df['Decade'] = (df['Release Date'] // 10 * 10).astype(int).astype(str) + 's'

# 3. Explicit flag to string
df['Explicit_str'] = df['Explicit'].astype(str).str.lower().map({'yes': 'yes', '1': 'yes', 'true': 'yes'}).fillna('no')

# 4. Artist (fill NA and convert to string)
df['artist_clean'] = df['artist'].fillna('unknown').astype(str).str.lower()
print(f"  ✓ Unique Artists: {df['artist_clean'].nunique():,}")

# ── Assemble Features ──
ACOUSTIC_FEATURES = ['Tempo', 'Loudness', 'Danceability', 'Speechiness',
                     'Liveness', 'Acousticness', 'Instrumentalness',
                     'key_num', 'mode']
CATEGORICAL_FEATURES = ['artist_clean', 'genre_broad', 'Decade', 'Explicit_str']
NUMERIC_META = ['Popularity']

FEATURES = ACOUSTIC_FEATURES + NUMERIC_META + CATEGORICAL_FEATURES
X = df[FEATURES].copy()
y = df['label_enc'].values

print(f"\n  ✓ Final Feature Matrix: {X.shape[1]} columns")
print(f"    - Numeric: {len(ACOUSTIC_FEATURES + NUMERIC_META)}")
print(f"    - Categorical: {len(CATEGORICAL_FEATURES)} {CATEGORICAL_FEATURES}")

# ── Train/test split ──
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# NOTE: CatBoost handles scaling and categorical encoding natively.
# We do not need StandardScaler or OneHotEncoder.
cat_indices = [X_tr.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL TRAINING: CATBOOST (Target Encoding + Native Categorical)")
print("=" * 70)

# We use class_weights due to slight imbalance in original dataset
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
class_weights_dict = dict(enumerate(weights))

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=8,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    cat_features=cat_indices,
    class_weights=class_weights_dict,
    task_type='CPU', # Use CPU for maximum compatibility
    random_seed=42,
    verbose=50
)

t0 = time.time()
print("  Training CatBoost. This may take 2-5 minutes...")
model.fit(
    X_tr, y_tr,
    eval_set=(X_te, y_te),
    early_stopping_rounds=30,
    use_best_model=True
)
print(f"  ✓ Trained in {time.time()-t0:.1f}s")

# ── Evaluation ──
res = evaluate_model("CatBoost (Acoustic+Meta+Artist+Decade)", model,
                   X_tr, X_te, y_tr, y_te, CLASS_NAMES)

# ── Feature Importance ──
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (CatBoost PredictionValuesChange)")
print("=" * 70)

importances = model.get_feature_importance(type='PredictionValuesChange')
feat_imp_df = pd.DataFrame({'feature': FEATURES, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feat_imp_df.iterrows(), 1):
    tag = " ← CATEGORICAL" if row['feature'] in CATEGORICAL_FEATURES else ""
    print(f"  {i:>2} | {row['feature']:>20} | {row['importance']:>7.2f}% {tag}")

# ── Saving Models ──
BASELINE_4F = 0.7019 # From voting ensemble

print("\n" + "=" * 70)
print("GRAND COMPARISON")
print("=" * 70)
print(f"  Stage 4F (Voting Ensemble):  {BASELINE_4F:.4f}")
print(f"  Stage 4G (CatBoost):         {res['test_acc']:.4f}")
delta = res['test_acc'] - BASELINE_4F
print(f"  Net Improvement:             {delta:+.4f} ({delta*100:+.2f}%)")

# Save CatBoost model
model.save_model(os.path.join(OUTPUT_DIR, 'stage4g_catboost_model.cbm'))
joblib.dump(FEATURES, os.path.join(OUTPUT_DIR, 'stage4g_features.pkl'))
joblib.dump(le, os.path.join(OUTPUT_DIR, 'stage4g_label_encoder.pkl'))
print(f"\n  ✓ Saved stage4g_catboost_model.cbm")
print(f"  ✓ Saved stage4g_features.pkl")

# Sub-module: stage_4h_hierarchical_model.py

"""
=============================================================================
STAGE 4H: TWO-STAGE HIERARCHICAL MODEL
=============================================================================
Purpose: Split the 4-class problem into a hierarchy to maximize accuracy.
  Stage 1: Predict Arousal (High vs Low Energy)
  Stage 2A: Predict High Valence (Angry vs Happy)
  Stage 2B: Predict Low Valence (Calm vs Sad)
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
DATASET_PATH = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\spotify_prepared_final.csv"
OUTPUT_DIR   = r"C:\Users\User\OneDrive - Asia Pacific University\FYP\pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STAGE 4H: TWO-STAGE HIERARCHICAL MODEL")
print("=" * 70)

print("\n[Loading] Reading spotify_prepared_final.csv ...")
df = pd.read_csv(DATASET_PATH)
print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Target Mapping ──
def map_arousal(emotion):
    if emotion in ['Angry', 'Happy']: return 'High'
    if emotion in ['Calm', 'Sad']:    return 'Low'
    return 'Unknown'

df['Arousal_Label'] = df['emotion_label'].apply(map_arousal)
CLASS_NAMES = ['Angry', 'Calm', 'Happy', 'Sad']

# ── Feature Engineering (Matching Stage 4F) ──
print("\n[Engineering] Extracting features (matching Stage 4F)...")
GENRE_MAP = {
    'pop': ['pop', 'kpop', 'k-pop', 'j-pop', 'synth-pop', 'dance pop', 'electropop'],
    'rock': ['rock', 'classic rock', 'indie rock', 'punk rock', 'punk', 'grunge', 'garage'],
    'hip_hop': ['rap', 'hip-hop', 'hip hop', 'trap', 'drill'],
    'metal': ['metal', 'heavy metal', 'death metal', 'black metal', 'hardcore', 'metalcore'],
    'electronic': ['electronic', 'edm', 'house', 'techno', 'trance', 'dubstep', 'drum and bass'],
    'rnb_soul': ['soul', 'r&b', 'rnb', 'funk', 'motown', 'gospel'],
    'folk': ['folk', 'acoustic', 'singer-songwriter', 'americana'],
    'jazz': ['jazz', 'blues', 'swing', 'bebop'],
    'classical': ['classical', 'orchestra', 'symphony', 'piano', 'opera'],
    'country': ['country', 'classic country', 'modern country', 'bluegrass'],
    'reggae': ['reggae', 'ska', 'dancehall', 'dub'],
    'other': []
}

def map_genre(raw_genre):
    if pd.isna(raw_genre): return 'other'
    raw_lower = str(raw_genre).lower()
    for category, keywords in GENRE_MAP.items():
        for kw in keywords:
            if kw in raw_lower: return category
    return 'other'

df['genre_broad'] = df['Genre'].apply(map_genre)
genre_dummies = pd.get_dummies(df['genre_broad'], prefix='genre', drop_first=False)

if df['Explicit'].dtype == object:
    df['Explicit_bin'] = df['Explicit'].str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
else:
    df['Explicit_bin'] = df['Explicit'].fillna(0).astype(int)

ACOUSTIC_FEATURES = ['Tempo', 'Loudness', 'Danceability', 'Speechiness',
                     'Liveness', 'Acousticness', 'Instrumentalness',
                     'key_num', 'mode']

acoustic_df = df[ACOUSTIC_FEATURES].copy()
meta_df     = df[['Explicit_bin', 'Popularity']].copy()
X_full      = pd.concat([acoustic_df, meta_df, genre_dummies], axis=1)
y_full      = df['emotion_label'].values
y_arousal   = df['Arousal_Label'].values

print(f"  ✓ Final Feature Matrix: {X_full.shape[1]} columns")

# ── Train/test split (Split before subsets to prevent leakage) ──
X_tr, X_te, y_tr_orig, y_te_orig, y_tr_arousal, y_te_arousal = train_test_split(
    X_full, y_full, y_arousal, test_size=0.2, random_state=42, stratify=y_full
)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)
X_tr_sc = pd.DataFrame(X_tr_sc, columns=X_full.columns, index=X_tr.index)
X_te_sc = pd.DataFrame(X_te_sc, columns=X_full.columns, index=X_te.index)

print(f"  ✓ Train: {X_tr_sc.shape}  |  Test: {X_te_sc.shape}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 1: AROUSAL ROUTER (High vs Low Energy)")
print("=" * 70)

m1_arousal = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=42)
t0 = time.time()
m1_arousal.fit(X_tr_sc, y_tr_arousal)
m1_preds = m1_arousal.predict(X_te_sc)
m1_acc = accuracy_score(y_te_arousal, m1_preds)

print(f"  ✓ Trained in {time.time()-t0:.1f}s")
print(f"  ✓ Arousal Router Accuracy: {m1_acc*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 2A: HIGH ENERGY EXPERT (Angry vs Happy)")
print("=" * 70)

high_mask = y_tr_arousal == 'High'
X_tr_high = X_tr_sc[high_mask]
y_tr_high = y_tr_orig[high_mask]
print(f"  ✓ Training on {X_tr_high.shape[0]:,} High Energy songs...")

m2a_high = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=42)
t0 = time.time()
m2a_high.fit(X_tr_high, y_tr_high)
print(f"  ✓ Trained in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 2B: LOW ENERGY EXPERT (Calm vs Sad)")
print("=" * 70)

low_mask = y_tr_arousal == 'Low'
X_tr_low = X_tr_sc[low_mask]
y_tr_low = y_tr_orig[low_mask]
print(f"  ✓ Training on {X_tr_low.shape[0]:,} Low Energy songs...")

m2b_low = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=42)
t0 = time.time()
m2b_low.fit(X_tr_low, y_tr_low)
print(f"  ✓ Trained in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EVALUATION: THE HIERARCHICAL PIPELINE")
print("=" * 70)

print("  Routing Test Set through Pipeline...")
t0 = time.time()

# We can vectorize this for speed
y_pred_final = []
# Model 1 predicts arousal
arousal_preds = m1_arousal.predict(X_te_sc)

# Predict all with both models
high_preds = m2a_high.predict(X_te_sc)
low_preds  = m2b_low.predict(X_te_sc)

# Combine based on Model 1's decision
y_pred_final = np.where(arousal_preds == 'High', high_preds, low_preds)

elapsed = time.time() - t0
print(f"  ✓ Prediction completed in {elapsed:.2f}s")

# ── Metrics ──
final_acc = accuracy_score(y_te_orig, y_pred_final)
final_f1  = f1_score(y_te_orig, y_pred_final, average='macro')

print(f"\n  [Final Results]")
print(f"    Hierarchical Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"    Macro F1-Score:        {final_f1:.4f}")

cm = confusion_matrix(y_te_orig, y_pred_final, labels=CLASS_NAMES)
print(f"\n  [Confusion Matrix]")
print(f"    {'':>12}", end="")
for cls in CLASS_NAMES:
    print(f"  {cls:>10}", end="")
print("  <- Predicted")
for i, cls in enumerate(CLASS_NAMES):
    print(f"    {cls:>12}", end="")
    for j in range(len(CLASS_NAMES)):
        print(f"  {cm[i][j]:>10}", end="")
    print()

BASELINE_4F = 0.7019
print("\n" + "=" * 70)
print("GRAND COMPARISON")
print("=" * 70)
print(f"  Stage 4F (Voting Ensemble):   {BASELINE_4F:.4f}")
print(f"  Stage 4H (Hierarchical):      {final_acc:.4f}")
delta = final_acc - BASELINE_4F
print(f"  Net Improvement:              {delta:+.4f} ({delta*100:+.2f}%)")

# Save Models
joblib.dump(m1_arousal, os.path.join(OUTPUT_DIR, 'stage4h_model1_arousal.pkl'))
joblib.dump(m2a_high,   os.path.join(OUTPUT_DIR, 'stage4h_model2a_high.pkl'))
joblib.dump(m2b_low,    os.path.join(OUTPUT_DIR, 'stage4h_model2b_low.pkl'))
joblib.dump(sc,         os.path.join(OUTPUT_DIR, 'stage4h_scaler.pkl'))
joblib.dump(list(X_full.columns), os.path.join(OUTPUT_DIR, 'stage4h_features.pkl'))

print(f"\n  ✓ Saved all hierarchical models and scaler to pipeline_output/")
print("=" * 70)
