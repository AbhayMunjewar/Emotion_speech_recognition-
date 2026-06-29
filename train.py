import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import os
import librosa

# ── 1. Feature Extraction (Richer Features: 400+) ────────────────────────
def extract_rich_features(audio, sr):
    try:
        features = []
        
        # 1. MFCCs mean + std + delta + delta2 → 40×4 = 160
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(mfcc.T, axis=0))
        features.extend(np.std(mfcc.T, axis=0))
        features.extend(np.mean(mfcc_delta.T, axis=0))
        features.extend(np.mean(mfcc_delta2.T, axis=0))

        # 2. Chroma mean + std → 12×2 = 24
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))

        # 3. Mel spectrogram mean + std → 128×2 = 256
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        features.extend(np.mean(mel.T, axis=0))
        features.extend(np.std(mel.T, axis=0))

        # 4. Spectral features (mean + std) → 20
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        features.extend([np.mean(spec_cent), np.std(spec_cent)])
        features.extend([np.mean(spec_bw), np.std(spec_bw)])
        features.extend([np.mean(spec_rolloff), np.std(spec_rolloff)])
        features.extend(np.mean(spec_contrast.T, axis=0))
        features.extend(np.std(spec_contrast.T, axis=0))

        # 5. Spectral flatness → 2
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        features.extend([np.mean(spec_flat), np.std(spec_flat)])

        # 6. Tonnetz (tonal centroid features) → 12
        harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        features.extend(np.mean(tonnetz.T, axis=0))
        features.extend(np.std(tonnetz.T, axis=0))

        # 7. ZCR + RMS → 4
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        features.extend([np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)])

        # 8. Pitch / F0 stats → 4
        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
        features.extend([np.mean(f0_clean), np.std(f0_clean)])
        features.extend([np.max(f0_clean) - np.min(f0_clean), np.mean(voiced)])

        return np.array(features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def load_or_extract_features():
    features_file = "X_features_rich_aug.npy"
    labels_file = "y_labels_rich_aug.npy"
    dataset_path = "dataset"
    
    if os.path.exists(features_file) and os.path.exists(labels_file):
        print(f"Loading cached richer augmented features from {features_file}...")
        return np.load(features_file), np.load(labels_file)
        
    print("Extracting 400+ richer features with Data Augmentation (4x dataset size). This will take ~10 minutes...")
    X, y = [], []
    count = 0
    for actor_folder in sorted(os.listdir(dataset_path)):
        actor_path = os.path.join(dataset_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
            
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                try:
                    emotion = int(file.split('-')[2]) - 1
                except:
                    continue
                
                # Load Audio
                try:
                    audio, sr = librosa.load(file_path, duration=3.0, offset=0.5)
                except Exception as e:
                    print(f"Failed to load {file_path}")
                    continue

                # 1. Original
                feat = extract_rich_features(audio, sr)
                if feat is not None:
                    X.append(feat)
                    y.append(emotion)
                
                # 2. Noise
                noise_audio = add_noise(audio)
                feat_noise = extract_rich_features(noise_audio, sr)
                if feat_noise is not None:
                    X.append(feat_noise)
                    y.append(emotion)
                
                # 3. Stretch
                stretch_audio = stretch(audio)
                feat_stretch = extract_rich_features(stretch_audio, sr)
                if feat_stretch is not None:
                    X.append(feat_stretch)
                    y.append(emotion)
                
                # 4. Pitch
                pitch_audio = pitch(audio, sr)
                feat_pitch = extract_rich_features(pitch_audio, sr)
                if feat_pitch is not None:
                    X.append(feat_pitch)
                    y.append(emotion)

                count += 1
                if count % 50 == 0:
                    print(f"Processed {count} original files (extracted {count * 4} samples)...")
                        
    X = np.array(X)
    y = np.array(y)
    np.save(features_file, X)
    np.save(labels_file, y)
    return X, y

X, y = load_or_extract_features()

print(f"X shape: {X.shape}")   # Expected: (1440, 482)
print(f"y shape: {y.shape}")   # Expected: (1440,)

# ── 2. Split -> train / test ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 80% train (1152 files), 20% test (288 files)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. Scale & Select Features ────────────────────────────
# MLP needs features on same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Using all {X_train.shape[1]} features (no feature selection needed for augmented dataset).")

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ── 4. Train Model ──────────────────────────────────────
print("\nTraining Ensemble (MLP + SVM + ExtraTrees + XGBoost) to reach 90%+...")

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate='adaptive',
    max_iter=800,
    random_state=42
)

svm = SVC(
    C=10.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42
)

et = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model = VotingClassifier(
    estimators=[('mlp', mlp), ('svm', svm), ('et', et), ('xgb', xgb)],
    voting='soft'
)

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
