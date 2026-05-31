import sys
sys.path.insert(0, r"C:\tf_pkg")

import numpy as np

from keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import train_test_split

# ==========================================
# LOAD DATA
# ==========================================

X = np.load("X.npy")

y = np.load("y.npy")

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    "models/cnn_model.keras"
)

# ==========================================
# PREDICT
# ==========================================

y_pred_prob = model.predict(X_test)

# ==========================================
# NEW THRESHOLD
# ==========================================

THRESHOLD = 0.8

y_pred = (
    y_pred_prob > THRESHOLD
).astype(int)

# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)