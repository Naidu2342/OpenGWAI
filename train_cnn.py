import sys
sys.path.insert(0, r"C:\tf_pkg")

from pathlib import Path

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"

from keras.models import Sequential

from keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)

from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from keras.optimizers import Adam

# ==========================================
# SETTINGS
# ==========================================

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42

EPOCHS = 25

BATCH_SIZE = 64

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

X = np.load("X.npy")

y = np.load("y.npy")

print(f"\nX shape: {X.shape}")

print(f"y shape: {y.shape}")

# ==========================================
# RESHAPE FOR CNN
# ==========================================

print("\nReshaping dataset for CNN...")

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

print(f"\nNew X shape: {X.shape}")

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain Samples: {len(X_train)}")

print(f"Test Samples: {len(X_test)}")

# ==========================================
# CLASS WEIGHTS
# ==========================================

print("\nComputing class weights...")

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass Weights:")

print(class_weights)

# ==========================================
# BUILD CNN MODEL
# ==========================================

print("\nBuilding CNN model...")

model = Sequential()

# ======================================
# CONV BLOCK 1
# ======================================

model.add(
    Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu",
        input_shape=(X.shape[1], 1)
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling1D(pool_size=2)
)

# ======================================
# CONV BLOCK 2
# ======================================

model.add(
    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling1D(pool_size=2)
)

# ======================================
# CONV BLOCK 3
# ======================================

model.add(
    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling1D(pool_size=2)
)

# ======================================
# FLATTEN
# ======================================

model.add(
    Flatten()
)

# ======================================
# DENSE
# ======================================

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dropout(0.4)
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dropout(0.3)
)

# ======================================
# OUTPUT
# ======================================

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# ==========================================
# COMPILE MODEL
# ==========================================

print("\nCompiling model...")

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:\n")

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# ==========================================
# EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

y_pred_prob = model.predict(X_test)

y_pred = (
    y_pred_prob > 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)

# ==========================================
# SAVE MODEL
# ==========================================

print("\nSaving model...")

model.save(
    MODEL_DIR / "cnn_model.keras"
)

# ==========================================
# SAVE METADATA
# ==========================================

metadata = {
    "window_size": int(X.shape[1]),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "test_accuracy": float(accuracy),
    "classes": {
        "0": "noise",
        "1": "wave"
    }
}

with open(
    MODEL_DIR / "cnn_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

print("\nModel Saved Successfully!")

print("\nSaved Files:")

print("- models/cnn_model.keras")

print("- models/cnn_metadata.json")

# ==========================================
# TRAINING CURVES
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# ==========================================
# LOSS CURVE
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("\nDONE")