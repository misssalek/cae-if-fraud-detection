# utils.py

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef




def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Type I Error Rate': fp / (tn + fp) if (tn + fp) > 0 else 0,
        'Type II Error Rate': fn / (tp + fn) if (tp + fn) > 0 else 0
    }
