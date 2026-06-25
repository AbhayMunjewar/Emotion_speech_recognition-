import os
import numpy as np
import librosa

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3.0, offset=0.5)
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
        features.append(np.mean(spec_cent))
        features.append(np.std(spec_cent))
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        features.extend(np.mean(spec_contrast.T, axis=0))   # 7
        features.extend(np.std(spec_contrast.T, axis=0))    # 7

        # 5. Spectral flatness → 2
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        features.append(np.mean(spec_flat))
        features.append(np.std(spec_flat))

        # 6. Tonnetz (tonal centroid features) → 12
        harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        features.extend(np.mean(tonnetz.T, axis=0))   # 6
        features.extend(np.std(tonnetz.T, axis=0))    # 6

        # 7. ZCR + RMS → 4
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # 8. Pitch / F0 stats → 4
        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
        features.append(np.mean(f0_clean))
        features.append(np.std(f0_clean))
        features.append(np.max(f0_clean) - np.min(f0_clean))  # pitch range
        features.append(np.mean(voiced))  # voiced fraction

        return np.array(features)

    except Exception as e:
        print(f"Error: {file_path} → {e}")
        return None

def build_dataset(data_path):
    X, y = [], []
    emotion_map = {
        1:'neutral', 2:'calm', 3:'happy', 4:'sad',
        5:'angry', 6:'fearful', 7:'disgust', 8:'surprised'
    }
    count = 0

    for actor_folder in sorted(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file_name in os.listdir(actor_path):
            if not file_name.endswith('.wav'):
                continue

            parts = file_name.split('-')
            emotion_code = int(parts[2])
            feat = extract_features(os.path.join(actor_path, file_name))

            if feat is not None:
                X.append(feat)
                y.append(emotion_code - 1)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} files...")

    X = np.array(X)
    y = np.array(y)
    print(f"\nFeature shape: {X.shape}")
    print(f"Total samples: {len(y)}")
    np.save("X_features_v2.npy", X)
    np.save("y_labels_v2.npy", y)
    print("Saved X_features_v2.npy and y_labels_v2.npy")

if __name__ == "__main__":
    build_dataset("dataset")