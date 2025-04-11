import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.models.iforest import IForest

from config import batch_size, train_step, SEED
from data_loader import load_data
from model import encoder, decoder, loss, gradiant, weights, biases, save_weights_and_biases, load_weights_and_biases
from utils import classification_metrics

import datetime
print('**************************************')
print('CAE_IF -', datetime.datetime.now())

X, y = load_data()
scaler = StandardScaler()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

best_tr_loss = float('inf')
best_fold = None
fold = 1

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    train_features = scaler.fit_transform(np.array(X_train))
    test_features = scaler.transform(np.array(X_test))
    train_features = np.clip(train_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    tx_train = tf.convert_to_tensor(train_features, dtype=tf.float32)
    tx_train = tf.reshape(tx_train, [tx_train.shape[0], 29, 1])
    tx_test = tf.convert_to_tensor(test_features, dtype=tf.float32)
    tx_test = tf.reshape(tx_test, [tx_test.shape[0], 29, 1])

    for step in range(train_step):
        for i in range(int(tx_train.shape[0] / batch_size) - 1):
            batch = tx_train[i * batch_size : (i + 1) * batch_size]
            gradiant(batch)
        tr_loss = loss(tx_train, decoder(encoder(tx_train), tx_train.shape[0]))
        print("step{:02d}:   loss:{:.06f}".format(step, tr_loss))

    if tr_loss < best_tr_loss:
        best_tr_loss = tr_loss
        best_fold = fold
        save_weights_and_biases(weights, biases, fold)
    
    fold += 1

print(f"Best model found at fold {best_fold} with loss {best_tr_loss:.6f}")
weights, biases = load_weights_and_biases(best_fold)

# Run Isolation Forest with encoder output
outlier_fraction = 0.002
all_auc_roc_scores = []
all_auprc_scores = []
all_metrics_list = []

for i in range(1, 6):
    print(f"Starting Iteration {i}...")
    
    iForest = IForest(contamination=outlier_fraction)
    test_pred = encoder(tx_test)
    train_pred = encoder(tx_train)
    
    iForest.fit(train_pred)
    test_scores = iForest.decision_function(test_pred)
    y_pred = iForest.predict(test_pred)
    
    auc_roc = roc_auc_score(y_test, test_scores)
    auprc = average_precision_score(y_test, test_scores)
    metrics = classification_metrics(y_test, y_pred)

    all_auc_roc_scores.append(auc_roc)
    all_auprc_scores.append(auprc)
    all_metrics_list.append(metrics)

    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 30)

avg_auc_roc = np.mean(all_auc_roc_scores)
avg_auprc = np.mean(all_auprc_scores)
avg_metrics = {metric: np.mean([m[metric] for m in all_metrics_list]) for metric in all_metrics_list[0].keys()}

print("\n=== Final Average Metrics Across Iterations ===")
print(f"Average AUC-ROC: {avg_auc_roc:.4f}")
print(f"Average AUPRC: {avg_auprc:.4f}")
for metric, value in avg_metrics.items():
    print(f"Average {metric}: {value:.4f}")
print('******************************************')
