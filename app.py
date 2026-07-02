import os
import tempfile
import numpy as np
import librosa
import joblib
import noisereduce as nr
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = "emotion_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Missing emotion_model.pkl or scaler.pkl. Please run train.py first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

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

def preprocess_audio(audio, sr, target_seconds=3.0):
    """
    Preprocess a chunk of microphone audio to match RAVDESS training conditions.
    """
    # 1. Gentle silence trim (top_db=20 is gentle, preserves speech)
    # We only trim if it's the very beginning/end of the full file, but this
    # function is now called per-chunk, so we might want to skip trimming here
    # or keep it light. Let's just peak normalize and pad/crop to exactly target_seconds.
    
    target_len = int(target_seconds * sr)
    if len(audio) < target_len:
        # Pad with silence (same as training)
        audio_final = np.pad(audio, (0, target_len - len(audio)), 'constant')
    else:
        # Crop to target_seconds (just in case)
        audio_final = audio[:target_len]

    # Peak normalize (same as what librosa.load does by default for RAVDESS files)
    if np.max(np.abs(audio_final)) > 0:
        audio_final = librosa.util.normalize(audio_final)

    return audio_final

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "uploaded_audio.wav")
        file.save(temp_path)

        # Load audio at 22050 Hz (librosa default, matches training)
        audio, sr = librosa.load(temp_path, sr=22050)

        # 1. Global noise reduction
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.6, stationary=True)

        # 2. Global gentle silence trim
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # 3. Sliding window analysis for long audio
        chunk_length = int(3.0 * sr)
        hop_length = int(1.5 * sr) # 1.5 seconds overlap

        all_probabilities = []

        # If audio is very short, just do one chunk
        if len(audio) < chunk_length:
            chunks = [audio]
        else:
            # Extract overlapping 3-second chunks
            chunks = []
            for i in range(0, len(audio) - chunk_length + 1, hop_length):
                chunks.append(audio[i:i + chunk_length])
            # If the last chunk doesn't perfectly align, grab the last 3 seconds
            if len(audio) > chunk_length and (len(audio) - chunk_length) % hop_length != 0:
                chunks.append(audio[-chunk_length:])

        for chunk in chunks:
            # Preprocess chunk (pad/crop, normalize)
            chunk_std = preprocess_audio(chunk, sr)

            # Extract features
            features = extract_rich_features(chunk_std, sr)
            if features is not None:
                # Scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                # Predict probabilities
                probs = model.predict_proba(features_scaled)[0]
                all_probabilities.append(probs)

        if not all_probabilities:
            return jsonify({"error": "Failed to extract features from audio"}), 500

        # Average the probabilities across all chunks
        avg_probabilities = np.mean(all_probabilities, axis=0)
        prediction = EMOTIONS[np.argmax(avg_probabilities)]

        # Map to dict
        prob_dict = {EMOTIONS[i]: float(avg_probabilities[i]) for i in range(len(EMOTIONS))}

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probabilities": prob_dict
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
