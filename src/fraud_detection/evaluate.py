# src/fraud_detection/evaluate.py
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report
)
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Calculates key metrics and returns them as a dictionary."""
    
    # Get predicted probabilities for the positive class (fraud)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classify based on the threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    # --- THIS IS THE KEY CHANGE ---
    # Calculate all the metrics we care about
    auc_pr = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f"Area Under Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")
    
    # Plotting the curve
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig("precision_recall_curve.png")
    print("Precision-Recall curve saved to precision_recall_curve.png")
    
    # Return all metrics in a dictionary
    return {
        'auc_pr': auc_pr,
        'f1_score': f1,
        'recall': recall,
        'precision': precision
    }