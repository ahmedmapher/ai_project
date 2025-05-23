import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found. Please ensure the file is in the correct directory.")
    exit()

print("--- First 5 rows of the dataset ---")
print(df.head())
print("\n--- Column names ---")
print(df.columns)
print(f"\n--- Size of dataset (rows, columns) ---: {df.shape}")
print(f"Number of records: {len(df)}")

sns.set_style("darkgrid")
class_counts = df['ACTION'].value_counts()
total_samples = len(df)

plt.figure(figsize=(8, 6))
ax = class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Access Control Distribution', fontweight='bold', fontsize=15)
plt.xlabel('Employee Actions', fontweight='bold', fontsize=14)
plt.ylabel('Number of Employee Records', fontweight='bold', fontsize=14)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

for i, count in enumerate(class_counts):
    percentage = (count / total_samples) * 100
    ax.text(i, count + (total_samples * 0.01), f'{percentage:.2f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.ylim(0, class_counts.max() * 1.15) 
plt.tight_layout()
plt.show()

y = df["ACTION"]
X = df.drop("ACTION", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model, X_train_data, X_test_data, y_train_data, y_test_data, model_name="Model"):

    print(f"\n--- Evaluating: {model_name} ---")

    model.fit(X_train_data, y_train_data)

    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]

    accuracy = accuracy_score(y_test_data, y_pred)
    roc_auc = roc_auc_score(y_test_data, y_pred_proba)
    cm = confusion_matrix(y_test_data, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test_data, y_pred))

    print("\nConfusion Matrix:")

    if len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP): {tp}")

        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

        precision_class1 = precision_score(y_test_data, y_pred, pos_label=1, zero_division=0)
        recall_class1 = recall_score(y_test_data, y_pred, pos_label=1, zero_division=0) 
        f1_class1 = f1_score(y_test_data, y_pred, pos_label=1, zero_division=0)
        fpr_class0 = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"\nMetrics for Positive Class (1):")
        print(f"  Precision (Class 1): {precision_class1:.4f}")
        print(f"  Recall (Sensitivity/TPR Class 1): {recall_class1:.4f}")
        print(f"  F1-Score (Class 1): {f1_class1:.4f}")
        print(f"  False Positive Rate (FPR for Class 0 behaving as positive): {fpr_class0:.4f}")

    else:
        print(cm)


    print("-" * 50)
    return y_pred, y_pred_proba

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("\nTraining RandomForest Classifier...")
y_pred_rf, y_pred_proba_rf = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, model_name="RandomForest")

lgbm_hyperparameters = {
    'learning_rate': 0.1, 
    'max_depth': 7,      
    'n_estimators': 200, 
    'random_state': 42,
    'class_weight': 'balanced' 
}
lgbm_model = LGBMClassifier(**lgbm_hyperparameters)
print("\nTraining LightGBM Classifier...")
y_pred_lgbm, y_pred_proba_lgbm = evaluate_model(lgbm_model, X_train_scaled, X_test_scaled, y_train, y_test, model_name="LightGBM")


print("\n--- Script Finished ---")
