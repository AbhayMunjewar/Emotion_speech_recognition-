import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load ───────────────────────────────────────────────
X = np.load("X_features_v2.npy")
y = np.load("y_labels_v2.npy")
print(f"Feature shape: {X.shape}")
print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── 2. Split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. Scale ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 4. Feature selection (remove noisy features) ─────────
selector = SelectKBest(f_classif, k=min(300, X_train_scaled.shape[1]))
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)
print(f"Selected top {X_train_sel.shape[1]} features from {X_train_scaled.shape[1]}")

# ── 5. Train multiple classifiers and ensemble ───────────
print("\n--- Training Individual Models ---\n")

# Model 1: Tuned MLP
print("Training MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.005,          # L2 regularization (prevents overfitting)
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    batch_size=64,
    verbose=False
)
mlp.fit(X_train_sel, y_train)
mlp_acc = accuracy_score(y_test, mlp.predict(X_test_sel))
print(f"  MLP Accuracy: {mlp_acc*100:.2f}%")

# Model 2: SVM with RBF kernel (often best for speech emotion)
print("Training SVM...")
svm = SVC(
    C=10.0,
    kernel='rbf',
    gamma='scale',
    probability=True,     # needed for soft voting
    random_state=42
)
svm.fit(X_train_sel, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_sel))
print(f"  SVM Accuracy: {svm_acc*100:.2f}%")

# Model 3: Gradient Boosting
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_sel, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test_sel))
print(f"  Gradient Boosting Accuracy: {gb_acc*100:.2f}%")

# ── 6. Soft Voting Ensemble ──────────────────────────────
print("\nTraining Ensemble (Soft Voting)...")
ensemble = VotingClassifier(
    estimators=[('mlp', mlp), ('svm', svm), ('gb', gb)],
    voting='soft'  # uses prediction probabilities
)
# VotingClassifier needs to be fit, but we already have trained models
# So we use a pre-fitted approach
ensemble.estimators_ = [mlp, svm, gb]
ensemble.le_ = None
ensemble.classes_ = np.unique(y_train)

# Manual soft voting
probs_mlp = mlp.predict_proba(X_test_sel)
probs_svm = svm.predict_proba(X_test_sel)
probs_gb = gb.predict_proba(X_test_sel)
probs_avg = (probs_mlp + probs_svm + probs_gb) / 3.0
y_pred_ensemble = np.argmax(probs_avg, axis=1)
ens_acc = accuracy_score(y_test, y_pred_ensemble)
print(f"  Ensemble Accuracy: {ens_acc*100:.2f}%")

# ── 7. Pick best model ───────────────────────────────────
results = {
    'MLP': (mlp_acc, mlp.predict(X_test_sel)),
    'SVM': (svm_acc, svm.predict(X_test_sel)),
    'Gradient Boosting': (gb_acc, gb.predict(X_test_sel)),
    'Ensemble': (ens_acc, y_pred_ensemble)
}

best_name = max(results, key=lambda k: results[k][0])
best_acc, best_pred = results[best_name]
print(f"\n{'='*50}")
print(f"Best Model: {best_name} → {best_acc*100:.2f}%")
print(f"{'='*50}")

# ── 8. Classification report for best model ──────────────
emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, best_pred, target_names=emotions, zero_division=0))

# ── 9. Confusion matrix ──────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=emotions, yticklabels=emotions, cmap='Blues')
plt.title(f"{best_name} Confusion Matrix (Accuracy: {best_acc*100:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("mlp_v2_confusion_matrix.png")
print("Confusion matrix saved!")

# ── 10. Save best model + pipeline ───────────────────────
joblib.dump(mlp, "emotion_mlp_v2.pkl")
joblib.dump(svm, "emotion_svm_v2.pkl")
joblib.dump(scaler, "scaler_v2.pkl")
joblib.dump(selector, "selector_v2.pkl")
print(f"\nModels saved!")
print(f"  emotion_mlp_v2.pkl")
print(f"  emotion_svm_v2.pkl")
print(f"  scaler_v2.pkl")
print(f"  selector_v2.pkl")
