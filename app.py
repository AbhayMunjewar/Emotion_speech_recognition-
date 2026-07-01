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
        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
        features.extend([np.mean(f0_clean), np.std(f0_clean)])
        features.extend([np.max(f0_clean) - np.min(f0_clean), np.mean(voiced)])

        return np.array(features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

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

        # 1. Load audio
        audio, sr = librosa.load(temp_path, sr=22050)

        # 2. Noise reduction
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.75)

        # 3. Trim silence aggressively (threshold of 30 dB)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

        # 4. Do NOT zero-pad to 3 seconds if shorter, just process the active speech.
        # But if it's too long, crop it to 3.0 seconds to match max training length.
        target_len = int(3.0 * sr)
        if len(audio_trimmed) > target_len:
            audio = audio_trimmed[:target_len]
        else:
            audio = audio_trimmed

        # 5. RMS Volume Normalization
        # Instead of max-peak normalization, scale so the RMS energy matches a standard value (e.g. 0.05)
        # This matches the average loudness of RAVDESS.
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio * (0.05 / rms)
            # Prevent clipping just in case
            audio = np.clip(audio, -1.0, 1.0)

        # Extract features
        features = extract_rich_features(audio, sr)
        if features is None:
            return jsonify({"error": "Failed to extract features from audio"}), 500

        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        prediction = EMOTIONS[np.argmax(probabilities)]

        # Map to dict
        prob_dict = {EMOTIONS[i]: float(probabilities[i]) for i in range(len(EMOTIONS))}

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
