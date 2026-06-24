import os
import librosa
import numpy as np

def extract_features(file_path, n_mfcc=40):
    """
    Load audio file and extract MFCCs + Chroma + Mel features.
    Returns: 1D numpy array of shape (180,)
    """
    try:
        # Load audio - 3 sec clip, skip first 0.5s (silence)
        audio, sr = librosa.load(file_path, duration=3.0, offset=0.5)
        
        # 1. MFCCs — 40 coefficients, averaged over time
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)        # shape: (40,)
        
        # 2. Chroma — pitch class energy
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)       # shape: (12,)
        
        # 3. Mel Spectrogram — perceptual frequency
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)             # shape: (128,)
        
        # Concatenate all → shape (180,)
        return np.concatenate([mfccs_mean, chroma_mean, mel_mean])
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    dataset_path = "dataset"
    features = []
    labels = []
    
    print(f"Scanning dataset directory: {dataset_path} ...")
    count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                # Assume RAVDESS format (e.g. 03-01-06-01-02-01-12.wav) where the 3rd part is the emotion
                try:
                    emotion = int(file.split("-")[2])
                except:
                    emotion = None
                    
                feature_vector = extract_features(file_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    if emotion is not None:
                        labels.append(emotion)
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} files...")
                        
    print(f"\nTotal successfully processed files: {count}")
    
    if count > 0:
        X = np.array(features)
        y = np.array(labels)
        print(f"Features array shape (X): {X.shape}")
        
        if len(y) == len(X):
            print(f"Labels array shape (y): {y.shape}")
            np.save("X_features.npy", X)
            np.save("y_labels.npy", y)
            print("Successfully saved data to 'X_features.npy' and 'y_labels.npy'")
