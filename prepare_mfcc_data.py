"""
Prepare MFCC time-series data for 1D CNN training.
Saves MFCC sequences (not averaged), actor IDs, and emotion labels.
"""
import os
import numpy as np
import librosa

def extract_mfcc_sequence(file_path, n_mfcc=40, max_len=130):
    """Extract MFCC time-series (not averaged — preserves temporal info)."""
    try:
        audio, sr = librosa.load(file_path, duration=3.0, offset=0.5)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        # Also get delta and delta-delta for richer features
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Stack: (n_mfcc*3, time_steps) -> transpose to (time_steps, n_mfcc*3)
        features = np.vstack([mfccs, delta, delta2]).T  # (time, 120)
        
        # Normalize per-file
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Pad or trim to fixed length
        if features.shape[0] < max_len:
            pad = max_len - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)), mode='constant')
        else:
            features = features[:max_len]
        
        return features  # shape: (max_len, 120)
    
    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None

def build_dataset(data_path):
    X, y, actors = [], [], []
    
    for actor_folder in sorted(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        
        # Extract actor number from folder name (e.g., "Actor_01" -> 0)
        actor_id = int(actor_folder.split('_')[1]) - 1
        
        for file_name in sorted(os.listdir(actor_path)):
            if not file_name.endswith('.wav'):
                continue
            
            parts = file_name.split('-')
            emotion_code = int(parts[2])
            file_path = os.path.join(actor_path, file_name)
            
            feat = extract_mfcc_sequence(file_path)
            if feat is not None:
                X.append(feat)
                y.append(emotion_code - 1)  # 0-indexed
                actors.append(actor_id)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    actors = np.array(actors)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}, actors={actors.shape}")
    print(f"Unique actors: {np.unique(actors)}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Samples per actor: {dict(zip(*np.unique(actors, return_counts=True)))}")
    
    np.save("X_mfcc_seq.npy", X)
    np.save("y_mfcc_seq.npy", y)
    np.save("actors_mfcc_seq.npy", actors)
    print("\nSaved X_mfcc_seq.npy, y_mfcc_seq.npy, actors_mfcc_seq.npy")

if __name__ == "__main__":
    build_dataset("dataset")
