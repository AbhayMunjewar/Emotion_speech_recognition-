import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ── 1. Load saved features ──────────────────────────────
X = np.load("X_features_0_7.npy")
y = np.load("y_labels_0_7.npy")

print(f"X shape: {X.shape}")   # (1440, 180)
print(f"y shape: {y.shape}")   # (1440,)

# ── 2. Split -> train / test ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 80% train (1152 files), 20% test (288 files)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. Scale features ───────────────────────────────────
# MLP needs features on same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. Train MLP model ──────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),   # 2 hidden layers
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=True                     # shows progress
)

print("\nTraining started...")
model.fit(X_train, y_train)
print("Training done!")

# ── 5. Evaluate ─────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")

emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=emotions))

# ── 6. Confusion Matrix ─────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=emotions,
            yticklabels=emotions,
            cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
# plt.show() # Commented out so the background task doesn't freeze waiting for you to close the image window
print("Confusion matrix saved!")

# ── 7. Save model + scaler ──────────────────────────────
joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved as emotion_model.pkl")
print("Scaler saved as scaler.pkl")
