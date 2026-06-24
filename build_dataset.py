import os
import numpy as np
import pandas as pd
from extract_features import extract_features

def build_dataset(data_path):
    """
    Walk RAVDESS folder, extract features + labels for every .wav file.
    Returns: X (features), y (emotion labels 0-7), file_paths
    """
    features_list = []
    labels = []
    file_paths = []
    
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    
    for actor_folder in sorted(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        
        for file_name in os.listdir(actor_path):
            if not file_name.endswith('.wav'):
                continue
            
            # Parse emotion from filename
            parts = file_name.split('-')
            emotion_code = int(parts[2])  # 3rd segment
            
            file_path = os.path.join(actor_path, file_name)
            feat = extract_features(file_path)
            
            if feat is not None:
                features_list.append(feat)
                labels.append(emotion_code - 1)  # 0-indexed (0-7)
                file_paths.append(file_path)
                # Print progress to console
                print(f"[+] {file_name} -> {emotion_map.get(emotion_code, 'Unknown')}")
    
    X = np.array(features_list)
    y = np.array(labels)
    
    print(f"\nDataset built: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, file_paths

if __name__ == "__main__":
    # Updated paths for your workspace
    X, y, paths = build_dataset("dataset")
    
    # Save the updated output
    np.save("X_features_0_7.npy", X)
    np.save("y_labels_0_7.npy", y)
    print("Saved X_features_0_7.npy and y_labels_0_7.npy")
