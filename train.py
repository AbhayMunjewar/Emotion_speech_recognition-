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
import noisereduce as nr

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
        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr, hop_length=1536)
        f0_clean = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
        features.extend([np.mean(f0_clean), np.std(f0_clean)])
        features.extend([np.max(f0_clean) - np.min(f0_clean), np.mean(voiced)])

        return np.array(features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# ── Data Augmentation Functions ────────────────────────────────────────────
def add_noise(data):
    """Add realistic background noise to simulate microphone recordings."""
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    noise = noise_amp * np.random.normal(size=data.shape[0])
    return data + noise

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def standardize_audio(audio, sr, target_seconds=3.0):
    """
    Standardize audio to exactly target_seconds.
    This MUST match the preprocessing in app.py's preprocess_audio().
    """
    # Gentle silence trim
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    
    # Safety: if trimming left less than 0.5s, use original
    min_samples = int(0.5 * sr)
    if len(audio_trimmed) < min_samples:
        audio_trimmed = audio
    
    # Pad or crop to exactly target_seconds
    target_len = int(target_seconds * sr)
    if len(audio_trimmed) < target_len:
        audio_trimmed = np.pad(audio_trimmed, (0, target_len - len(audio_trimmed)), 'constant')
    else:
        audio_trimmed = audio_trimmed[:target_len]
    
    # Peak normalize
    audio_trimmed = librosa.util.normalize(audio_trimmed)
    
    return audio_trimmed

def load_or_extract_features():
    # Use v3 cache to force re-extraction with the new pipeline
    features_file = "X_features_v3.npy"
    labels_file = "y_labels_v3.npy"
    dataset_path = "dataset"
    
    if os.path.exists(features_file) and os.path.exists(labels_file):
        print(f"Loading cached features from {features_file}...")
        return np.load(features_file), np.load(labels_file)
        
    print("Extracting features with augmentation (6x per sample). This will take ~15 minutes...")
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
                    audio_raw, sr = librosa.load(file_path, sr=22050)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
                    continue

                # 1. Original (standardized exactly like app.py)
                audio_std = standardize_audio(audio_raw, sr)
                feat = extract_rich_features(audio_std, sr)
                if feat is not None:
                    X.append(feat)
                    y.append(emotion)
                
                # 2. With noise (simulates microphone recordings)
                noise_audio = add_noise(audio_raw)
                noise_std = standardize_audio(noise_audio, sr)
                feat_noise = extract_rich_features(noise_std, sr)
                if feat_noise is not None:
                    X.append(feat_noise)
                    y.append(emotion)
                
                # 3. Stretch (slower)
                stretch_audio = stretch(audio_raw, rate=0.8)
                stretch_std = standardize_audio(stretch_audio, sr)
                feat_stretch = extract_rich_features(stretch_std, sr)
                if feat_stretch is not None:
                    X.append(feat_stretch)
                    y.append(emotion)
                
                # 4. Stretch (faster)
                stretch_audio2 = stretch(audio_raw, rate=1.2)
                stretch_std2 = standardize_audio(stretch_audio2, sr)
                feat_stretch2 = extract_rich_features(stretch_std2, sr)
                if feat_stretch2 is not None:
                    X.append(feat_stretch2)
                    y.append(emotion)
                
                # 5. Pitch up
                pitch_audio = pitch(audio_raw, sr, pitch_factor=2.0)
                pitch_std = standardize_audio(pitch_audio, sr)
                feat_pitch = extract_rich_features(pitch_std, sr)
                if feat_pitch is not None:
                    X.append(feat_pitch)
                    y.append(emotion)
                
                # 6. Pitch down
                pitch_audio2 = pitch(audio_raw, sr, pitch_factor=-2.0)
                pitch_std2 = standardize_audio(pitch_audio2, sr)
                feat_pitch2 = extract_rich_features(pitch_std2, sr)
                if feat_pitch2 is not None:
                    X.append(feat_pitch2)
                    y.append(emotion)

                count += 1
                if count % 50 == 0:
                    print(f"Processed {count} files ({count * 6} augmented samples)...")
                        
    X = np.array(X)
    y = np.array(y)
    np.save(features_file, X)
    np.save(labels_file, y)
    return X, y

X, y = load_or_extract_features()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ── 2. Split -> train / test ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 3. Scale Features ────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Using all {X_train.shape[1]} features.")

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ── 4. Train Model ──────────────────────────────────────
print("\nTraining Ensemble (MLP + SVM + ExtraTrees + XGBoost)...")

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.005,
    learning_rate='adaptive',
    max_iter=1000,
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
print("Confusion matrix saved!")

# ── 7. Save model + scaler ──────────────────────────────
joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved as emotion_model.pkl")
print("Scaler saved as scaler.pkl")
