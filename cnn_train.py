"""
1D CNN + BiLSTM for Speech Emotion Recognition on RAVDESS.
Uses MFCC time-series (not spectrograms) — proven to work on small datasets.
Actor-grouped splits ensure the model learns emotions, not voice identity.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load data ─────────────────────────────────────────
X = np.load("X_mfcc_seq.npy")      # (N, 130, 120)
y = np.load("y_mfcc_seq.npy")      # (N,)
actors = np.load("actors_mfcc_seq.npy")  # (N,)

print(f"X shape: {X.shape}")
print(f"y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"actors: {np.unique(actors)}")

# ── 2. Actor-grouped split ───────────────────────────────
# Critical: same actor must NOT appear in train AND val/test
# Actor 0-17 (18 actors) → train
# Actor 18-19 (2 actors) → val
# Actor 20-23 (4 actors) → test

train_mask = actors <= 17
val_mask = (actors >= 18) & (actors <= 19)
test_mask = actors >= 20

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"\nTrain: {len(X_train)} (actors 1-18)")
print(f"Val:   {len(X_val)} (actors 19-20)")
print(f"Test:  {len(X_test)} (actors 21-24)")
print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Val labels:   {dict(zip(*np.unique(y_val, return_counts=True)))}")

# ── 3. Oversample minority classes ───────────────────────
unique, counts = np.unique(y_train, return_counts=True)
max_count = counts.max()

X_parts, y_parts = [X_train], [y_train]
for cls, cnt in zip(unique, counts):
    if cnt < max_count:
        deficit = max_count - cnt
        cls_idx = np.where(y_train == cls)[0]
        extra = np.random.choice(cls_idx, size=deficit, replace=True)
        X_parts.append(X_train[extra])
        y_parts.append(y_train[extra])

X_train_bal = np.concatenate(X_parts)
y_train_bal = np.concatenate(y_parts)
print(f"\nBalanced train: {len(X_train_bal)} samples")

# ── 4. Mild augmentation ─────────────────────────────────
def augment(x, label):
    # Small noise
    if tf.random.uniform(()) > 0.5:
        x = x + tf.random.normal(tf.shape(x), stddev=0.05)
    # Small time shift
    if tf.random.uniform(()) > 0.5:
        shift = tf.random.uniform((), -5, 5, dtype=tf.int32)
        x = tf.roll(x, shift=shift, axis=0)
    return x, label

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train_bal, y_train_bal))
train_ds = train_ds.shuffle(len(X_train_bal)).map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ── 5. 1D CNN + BiLSTM model ─────────────────────────────
# Much fewer parameters than 2D CNN on spectrograms
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),  # (130, 120)
    
    # 1D Conv blocks — extract local temporal patterns
    tf.keras.layers.Conv1D(64, 5, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv1D(128, 5, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv1D(128, 3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.3),
    
    # BiLSTM — capture temporal dependencies
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    tf.keras.layers.Dropout(0.4),
    
    # Classifier
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.summary()

# ── 6. Compile ────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── 7. Train ──────────────────────────────────────────────
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=8,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        )
    ]
)

# ── 8. Evaluate ───────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=emotions, zero_division=0))

# ── 9. Confusion matrix ───────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=emotions, yticklabels=emotions, cmap='Blues')
plt.title(f"1D CNN+LSTM Confusion Matrix (Accuracy: {acc*100:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png")

# ── 10. Training curves ──────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png")

# ── 11. Save ──────────────────────────────────────────────
model.save("emotion_cnn_model.keras")
print("Saved emotion_cnn_model.keras")