import os
import numpy as np
import librosa

def extract_melspectrogram(file_path, n_mels=128, max_len=128):
    try:
        audio, sr = librosa.load(file_path, duration=3.0, offset=0.5)
        
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, 
            n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize PER FILE (each file normalized independently)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        
        # Pad or trim to fixed width
        if mel_db.shape[1] < max_len:
            pad = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]
        
        return mel_db  # shape: (128, 128)
    
    except Exception as e:
        print(f"Error: {file_path} → {e}")
        return None

def build_cnn_dataset(data_path):
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
            file_path = os.path.join(actor_path, file_name)
            
            mel = extract_melspectrogram(file_path)
            if mel is not None:
                X.append(mel)
                y.append(emotion_code - 1)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} files...")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"X min: {X.min():.3f} | max: {X.max():.3f} | mean: {X.mean():.3f}")
    
    np.save("X_cnn.npy", X)
    np.save("y_cnn.npy", y)
    print("Saved X_cnn.npy and y_cnn.npy")

if __name__ == "__main__":
    build_cnn_dataset("dataset")