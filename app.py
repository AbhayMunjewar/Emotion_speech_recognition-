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

def preprocess_audio(audio, sr):
    """
    Preprocess microphone audio to match RAVDESS training conditions.
    This function must produce audio that looks like what train.py feeds
    into extract_rich_features().
    """
    # 1. Light noise reduction to remove mic static
    #    (RAVDESS was recorded in a studio with no background noise)
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.6, stationary=True)

    # 2. Gentle silence trim (top_db=20 is gentle, preserves speech)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

    # Safety: if trimming left less than 0.5s, use original audio
    min_samples = int(0.5 * sr)
    if len(audio_trimmed) < min_samples:
        audio_trimmed = audio

    # 3. Ensure exactly 3.0 seconds to match training data length
    target_len = int(3.0 * sr)  # 66150 samples
    if len(audio_trimmed) < target_len:
        # Pad with silence (same as training)
        audio_final = np.pad(audio_trimmed, (0, target_len - len(audio_trimmed)), 'constant')
    else:
        # Crop to 3.0 seconds (same as training)
        audio_final = audio_trimmed[:target_len]

    # 4. Peak normalize (same as what librosa.load does by default for RAVDESS files)
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

        # Preprocess to match training conditions
        audio = preprocess_audio(audio, sr)

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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
