// Application States
let mediaStream = null;
let audioContext = null;
let scriptProcessor = null;
let audioBuffers = [];
let recordingLength = 0;
let isRecording = false;
let recordTimer = null;
let recordingDuration = 0; // in seconds
let activeAudioBlob = null;

// DOM Elements
const btnRecord = document.getElementById("btn-record");
const recordText = document.getElementById("record-text");
const visualizerContainer = document.getElementById("visualizer-container");
const waveformCanvas = document.getElementById("waveform-canvas");
const canvasCtx = waveformCanvas.getContext("2d");
const timerDisplay = document.getElementById("timer");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileInfo = document.getElementById("file-info");
const fileNameDisplay = document.getElementById("file-name");
const btnRemoveFile = document.getElementById("btn-remove-file");

const btnAnalyze = document.getElementById("btn-analyze");
const emptyState = document.getElementById("empty-state");
const loader = document.getElementById("analysis-loader");
const resultsContainer = document.getElementById("results-container");

const predictionEmoji = document.getElementById("prediction-emoji");
const predictionValue = document.getElementById("prediction-value");
const barsList = document.getElementById("bars-list");

// Emoji map for emotions
const EMOJI_MAP = {
    neutral: "😐",
    calm: "😌",
    happy: "😊",
    sad: "😢",
    angry: "😡",
    fearful: "😨",
    disgust: "🤢",
    surprised: "😲"
};

// Color map for emotions
const COLOR_MAP = {
    neutral: "#718096",
    calm: "#4299e1",
    happy: "#ecc94b",
    sad: "#3182ce",
    angry: "#e53e3e",
    fearful: "#805ad5",
    disgust: "#38a169",
    surprised: "#dd6b20"
};

// Set canvas size
function resizeCanvas() {
    waveformCanvas.width = visualizerContainer.clientWidth - 32;
    waveformCanvas.height = 60;
}
window.addEventListener("resize", resizeCanvas);

// ── 1. Audio Recording & Encoding Logic ─────────────────────────────────

async function startRecording() {
    audioBuffers = [];
    recordingLength = 0;
    recordingDuration = 0;
    isRecording = true;
    activeAudioBlob = null;
    btnAnalyze.disabled = true;

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        
        // Show Visualizer
        visualizerContainer.classList.remove("hidden");
        resizeCanvas();
        
        // Initialize AudioContext
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(mediaStream);
        
        // Setup processor node (older API but reliable client-side wav recording)
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        
        scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const inputBuffer = e.inputBuffer.getChannelData(0);
            
            // Clone buffer data
            audioBuffers.push(new Float32Array(inputBuffer));
            recordingLength += inputBuffer.length;
            
            // Draw waveform
            drawWaveform(inputBuffer);
        };

        // UI Updates
        btnRecord.classList.add("recording");
        recordText.textContent = "Stop Recording";
        
        // Timer
        timerDisplay.textContent = "00:00";
        recordTimer = setInterval(() => {
            recordingDuration++;
            const secs = String(recordingDuration % 60).padStart(2, '0');
            const mins = String(Math.floor(recordingDuration / 60)).padStart(2, '0');
            timerDisplay.textContent = `${mins}:${secs}`;
            
            // Automatically stop recording after 4 seconds (safe length)
            if (recordingDuration >= 4) {
                stopRecording();
            }
        }, 1000);

    } catch (err) {
        console.error("Failed to access microphone:", err);
        alert("Microphone access denied or unavailable.");
        cleanupRecordingState();
    }
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    // UI Updates
    btnRecord.classList.remove("recording");
    recordText.textContent = "Record Again";
    clearInterval(recordTimer);

    // Stop streams
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (scriptProcessor) {
        scriptProcessor.disconnect();
    }
    if (audioContext) {
        audioContext.close();
    }

    // Convert buffers to standard 16-bit PCM WAV
    const mergedBuffer = mergeBuffers(audioBuffers, recordingLength);
    const sampleRate = audioContext.sampleRate;
    activeAudioBlob = encodeWAV(mergedBuffer, sampleRate);
    
    // Enable Analyze Button
    btnAnalyze.disabled = false;
}

function cleanupRecordingState() {
    isRecording = false;
    btnRecord.classList.remove("recording");
    recordText.textContent = "Start Recording";
    clearInterval(recordTimer);
    visualizerContainer.classList.add("hidden");
}

// Draw basic dynamic oscilloscope
function drawWaveform(buffer) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#ef4057';
    canvasCtx.beginPath();
    
    const sliceWidth = waveformCanvas.width / buffer.length;
    let x = 0;
    
    for (let i = 0; i < buffer.length; i++) {
        const v = buffer[i] * 1.5; // Gain factor for visual contrast
        const y = (v + 1) * waveformCanvas.height / 2;
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        x += sliceWidth;
    }
    
    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();
}

// Merges floating point array buffers
function mergeBuffers(channelBuffer, recordingLength) {
    const result = new Float32Array(recordingLength);
    let offset = 0;
    for (let i = 0; i < channelBuffer.length; i++) {
        result.set(channelBuffer[i], offset);
        offset += channelBuffer[i].length;
    }
    return result;
}

// Pure JS WAV container encoder
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw PCM) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);
    
    // Write samples
    floatTo16BitPCM(view, 44, samples);
    
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

// ── 2. Drag & Drop File Upload Handling ─────────────────────────────────

// Triggers browse dialog
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleSelectedFile(e.target.files[0]);
    }
});

// Drag enter/over highlight
["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    }, false);
});

// Drag leave remove highlight
["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
    }, false);
});

// Handle drop
dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt.files.length > 0) {
        handleSelectedFile(dt.files[0]);
    }
});

function handleSelectedFile(file) {
    if (!file.name.endsWith(".wav")) {
        alert("Please upload a WAV audio file.");
        return;
    }
    
    // Cleanup recorder state if active
    cleanupRecordingState();
    
    activeAudioBlob = file;
    fileNameDisplay.textContent = file.name;
    fileInfo.classList.remove("hidden");
    dropZone.classList.add("hidden");
    
    btnAnalyze.disabled = false;
}

btnRemoveFile.addEventListener("click", () => {
    activeAudioBlob = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    dropZone.classList.remove("hidden");
    btnAnalyze.disabled = true;
});

// ── 3. Predictions and Results Spectrum ─────────────────────────────────

btnRecord.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

btnAnalyze.addEventListener("click", async () => {
    if (!activeAudioBlob) return;

    // Show Loader
    emptyState.classList.add("hidden");
    resultsContainer.classList.add("hidden");
    loader.classList.remove("hidden");
    btnAnalyze.disabled = true;

    const formData = new FormData();
    formData.append("file", activeAudioBlob, "voice_sample.wav");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        loader.classList.add("hidden");
        btnAnalyze.disabled = false;

        if (data.status === "success") {
            displayResults(data.prediction, data.probabilities);
        } else {
            alert(`Analysis failed: ${data.error || "Unknown Error"}`);
            emptyState.classList.remove("hidden");
        }

    } catch (err) {
        console.error("Prediction Error:", err);
        alert("Server communication failed.");
        loader.classList.add("hidden");
        emptyState.classList.remove("hidden");
        btnAnalyze.disabled = false;
    }
});

function displayResults(prediction, probabilities) {
    // Set Emotion Text and Accent Color
    predictionValue.textContent = prediction;
    predictionEmoji.textContent = EMOJI_MAP[prediction] || "😐";
    
    const accentColor = COLOR_MAP[prediction] || "#e94057";
    predictionValue.style.backgroundImage = `linear-gradient(90deg, ${accentColor} 0%, #ffffff 100%)`;

    // Render Progress Bars
    barsList.innerHTML = "";
    
    // Sort emotions by probability value
    const sortedEmotions = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedEmotions.forEach(([emotion, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "bar-row";
        
        row.innerHTML = `
            <div class="bar-info">
                <span class="bar-name">${emotion} ${EMOJI_MAP[emotion] || ""}</span>
                <span class="bar-val">${pct}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: 0%; background: ${COLOR_MAP[emotion] || '#e94057'}"></div>
            </div>
        `;
        
        barsList.appendChild(row);
        
        // Trigger smooth width transition on next animation frame
        requestAnimationFrame(() => {
            row.querySelector(".bar-fill").style.width = `${pct}%`;
        });
    });

    resultsContainer.classList.remove("hidden");
}
