# Real-Time AI-Generated Answer Detection System (POC)

A sophisticated prototype to detect whether a candidate's spoken answer is AI-generated during live interviews. Features real-time speech-to-text, ensemble ML classification, and visual analytics.

## 🎯 Features

- **Real-Time Speech Recognition**: Dual STT support (Web Speech API + Faster-Whisper)
- **Ensemble ML Detection**: 5 models voting for robust AI detection
- **Debug Mode**: Analyze predictions with similar training samples
- **Visual Analytics**: Live line chart showing AI-likelihood trends
- **Text Input Fallback**: Manual text entry for testing without audio

## 📁 Folder Structure

```
project-root/
├── frontend/
│   ├── src/
│   │   └── components/
│   │       └── App.jsx          # Main React component with UI
│   ├── index.html
│   └── package.json
│
└── backend/
    ├── app.py                   # FastAPI server with all endpoints
    ├── detector/
    │   ├── __init__.py
    │   ├── classifier.py        # Detector class with predict/debug methods
    │   └── model/
    │       ├── clf.pkl          # Trained ensemble model (generated)
    │       └── debug_data.pkl   # Training data for similarity search (generated)
    ├── train.py                 # Training script with 240 samples
    ├── test_detector.py         # Automated & interactive testing
    └── requirements.txt
```

## 🛠️ Tech Stack

### Frontend
- **React 18.3.1** + **Vite 5.2.0** for fast development
- **Web Speech API** for browser-native speech recognition
- **MediaRecorder API** for audio capture (Whisper mode)
- **Tailwind-style CSS** for responsive UI

### Backend
- **FastAPI 0.115.0** + **uvicorn** for async API server
- **Faster-Whisper** (base model, CPU int8) for accurate transcription
- **sentence-transformers** (all-MiniLM-L6-v2) for 384-dim embeddings
- **scikit-learn** ensemble (5 classifiers with soft voting)

### ML Pipeline
The detection uses an ensemble of 5 models:
1. **Logistic Regression** (C=1.0)
2. **Random Forest** (100 estimators)
3. **SVM** with RBF kernel (probability=True)
4. **Gradient Boosting** (100 estimators)
5. **MLP Neural Network** (128, 64 hidden layers)

## 📊 Training Dataset

The model is trained on **240 carefully curated samples**:

| Category | Count | Description |
|----------|-------|-------------|
| Base Human | 100 | Natural, casual interview responses |
| Hard Human | 15 | Articulate but genuinely human responses |
| Base AI | 100 | Typical ChatGPT/AI-style answers |
| Hard AI | 15 | Subtle AI patterns, less obvious markers |
| Ambiguous | 10 | Edge cases for robustness |

### Expected Accuracy
- **5-Fold Cross-Validation**: ~85-92% accuracy
- **Human text scores**: 0.1 - 0.4 (low AI likelihood)
- **AI text scores**: 0.7 - 0.95 (high AI likelihood)

##  Quick Start (Windows)

### 1. Backend Setup

```cmd
cd backend
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt

:: Install CPU-only PyTorch (fixes Windows DLL errors)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

:: Train the ensemble model (required!)
python train.py
```

### 2. Start Backend Server

```cmd
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup

```cmd
cd ..\frontend
npm install
npm run dev
```

Open http://localhost:5173 in Chrome.

##  API Endpoints

### Health Check
```
GET /health
Response: { "status": "ok" }
```

### Detect AI Content
```
POST /detect
Body: { "text": "Your text here..." }
Response: {
  "score": 0.73,
  "flag": true,
  "explanation": "AI-like probability: 0.73; length words: 45"
}
```

### Debug Detection (with analysis)
```
POST /detect/debug
Body: { "text": "Your text here..." }
Response: {
  "score": 0.73,
  "flag": true,
  "explanation": "...",
  "model_votes": {
    "logistic_regression": 0.68,
    "random_forest": 0.75,
    "svm": 0.71,
    "gradient_boosting": 0.78,
    "mlp": 0.72
  },
  "confidence_level": "high",
  "similar_samples": [
    {"text": "Similar training text...", "label": "ai", "similarity": 0.89}
  ]
}
```

### Speech-to-Text (Whisper)
```
POST /stt
Body: FormData with 'audio' file (webm/wav)
Response: { "transcript": "Transcribed text..." }
```

## 🧪 Testing

### Automated Test Suite
```cmd
cd backend
python test_detector.py
```

Runs 14 predefined test cases covering:
- Obvious human responses
- Obvious AI-generated content
- Edge cases and ambiguous text

### Interactive Testing
```cmd
python test_detector.py --interactive
```

Type any text to get real-time detection with debug info.

### Quick API Test
```cmd
python -c "import requests; print(requests.post('http://localhost:8000/detect', json={'text':'As an AI language model, I cannot perform actions.'}).json())"
```

## 🎨 UI Features

### Recording Modes
- **WebSpeech**: Browser-native, fast but less accurate
- **Whisper**: Server-side processing, more accurate

### Score Visualization
- **Green Badge** (< 0.4): Likely human
- **Orange Badge** (0.4 - 0.7): Uncertain
- **Red Badge** (> 0.7): Likely AI-generated

### Line Chart
- Real-time plotting of detection scores (0-1 scale)
- Tracks last 20 detection results
- Visual trend analysis

## 🔧 Troubleshooting

### Windows DLL Error
```
OSError: [WinError 1114] ... c10.dll
```
**Fix**: Install CPU-only PyTorch:
```cmd
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
```

### All Scores ~0.5-0.6
**Cause**: Model not trained with full dataset
**Fix**: Re-run `python train.py` and restart backend

### Whisper Model Download Slow
First run downloads ~150MB base model. Subsequent runs use cached model.

### Web Speech API Not Working
- Use Chrome/Edge (not Firefox)
- Allow microphone permissions
- Check HTTPS or localhost only

## 📈 Future Improvements

- [ ] Add more training samples from real interview data
- [ ] Fine-tune transformer model for domain-specific detection
- [ ] Add speaker diarization for multi-person interviews
- [ ] Implement confidence calibration
- [ ] Add WebSocket for true real-time streaming
- [ ] Export detection reports (PDF/CSV)

## 📝 Version History

### v2.0 (Current)
- ✅ Ensemble ML with 5 classifiers (LR, RF, SVM, GB, MLP)
- ✅ 240 training samples (100 human + 100 AI + 40 edge cases)
- ✅ 5-fold cross-validation during training
- ✅ Debug endpoint with similar sample analysis
- ✅ Faster-Whisper integration for accurate STT
- ✅ Individual model vote visualization
- ✅ Confidence level indicators (low/medium/high)
- ✅ Automated test suite (`test_detector.py`)
- ✅ Text input fallback for manual testing
- ✅ Fixed line chart Y-axis (0-1 scale)

### v1.0 (Initial)
- Basic logistic regression classifier
- 4-sample training dataset
- Web Speech API only
- Simple score display

## 📄 License

MIT License - Free for educational and commercial use.
