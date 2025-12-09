import asyncio
import io
import os
import tempfile
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from detector.classifier import get_detector

class DetectRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text to analyze for AI-generated content",
        examples=[
            "I really enjoyed working on that project and learned a lot.",
            "As an AI language model, I can provide a comprehensive analysis."
        ]
    )

app = FastAPI(
    title="AI Answer Detector API",
    version="2.0.0",
    description="Detect AI-generated content in text using ensemble ML models. Supports whole-text detection, sentence-level analysis, and debug mode."
)

# Allow local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton detector
_detector = None
_detector_lock = asyncio.Lock()

async def get_detector_async():
    global _detector
    async with _detector_lock:
        if _detector is None:
            _detector = await get_detector()
    return _detector

# Lazy singleton Faster-Whisper model
_whisper_model = None
_whisper_lock = asyncio.Lock()

async def get_whisper_model():
    global _whisper_model
    async with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            # Use "base" for balance of speed/accuracy on CPU; "tiny" for faster
            _whisper_model = await asyncio.to_thread(
                WhisperModel, "base", device="cpu", compute_type="int8"
            )
    return _whisper_model

@app.get("/health", summary="Health Check", tags=["System"])
async def health():
    """Check if the API server is running."""
    return {"status": "ok"}

@app.post("/detect", summary="Detect AI Content", tags=["Detection"])
async def detect(payload: DetectRequest):
    """
    Detect if text is AI-generated using ensemble ML models.
    
    Returns a single score (0-1) where:
    - 0.0-0.4: Likely human-written
    - 0.4-0.7: Uncertain
    - 0.7-1.0: Likely AI-generated
    
    **Example Request:**
    ```json
    {
      "text": "As an AI language model, I cannot provide opinions."
    }
    ```
    
    **Example Response:**
    ```json
    {
      "score": 0.85,
      "flag": true,
      "explanation": "Ensemble score: 0.85; model: VotingClassifier; words: 9"
    }
    ```
    """
    detector = await get_detector_async()
    result = await detector.predict(payload.text)
    return result

@app.post("/detect/debug", summary="Debug Detection", tags=["Detection"])
async def detect_debug(payload: DetectRequest):
    """
    Debug mode: Get detailed analysis with individual model votes and similar training samples.
    
    Useful for understanding why the model made a particular prediction.
    
    **Returns:**
    - Individual votes from each of the 5 models (LR, RF, SVM, GB, MLP)
    - Confidence level (low/medium/high)
    - Top 5 most similar training samples with similarity scores
    
    **Example Response:**
    ```json
    {
      "score": 0.73,
      "flag": true,
      "explanation": "AI prob: 0.730, Human prob: 0.270, words: 12",
      "confidence": "high",
      "model_votes": {
        "LR": 0.68,
        "RF": 0.75,
        "SVM": 0.71,
        "GB": 0.78,
        "MLP": 0.72
      },
      "similar_samples": [
        {"text": "Similar training text...", "label": "AI", "similarity": 0.89}
      ]
    }
    ```
    """
    detector = await get_detector_async()
    result = await detector.predict_debug(payload.text)
    return result

@app.post("/detect/sentences", summary="Sentence-Level Detection", tags=["Detection"])
async def detect_sentences(payload: DetectRequest):
    """
    Analyze text sentence-by-sentence to detect mixed human/AI content.
    
    Perfect for identifying which parts of a text are AI-generated when content is partially written by AI.
    
    **Example Request:**
    ```json
    {
      "text": "I worked hard on this project. As an AI language model, I provide solutions. My team helped a lot."
    }
    ```
    
    **Example Response:**
    ```json
    {
      "overall_score": 0.623,
      "flag": true,
      "total_sentences": 3,
      "human_sentences": 2,
      "ai_sentences": 1,
      "uncertain_sentences": 0,
      "summary": "MIXED: 2 human, 1 AI, 0 uncertain",
      "sentences": [
        {"text": "I worked hard on this project.", "score": 0.245, "label": "Human"},
        {"text": "As an AI language model, I provide solutions.", "score": 0.891, "label": "AI"},
        {"text": "My team helped a lot.", "score": 0.312, "label": "Human"}
      ]
    }
    ```
    
    Each sentence is classified as:
    - **Human**: score < 0.35
    - **AI**: score > 0.65
    - **Uncertain**: score between 0.35 and 0.65
    """
    detector = await get_detector_async()
    result = await detector.predict_sentences(payload.text)
    return result

@app.post("/stt", summary="Speech to Text", tags=["STT"])
async def stt(audio: UploadFile = File(..., description="Audio file (wav, webm, mp3, etc.)")):
    """
    Convert speech audio to text using Faster-Whisper (base model).
    
    **Supported formats:** wav, webm, mp3, m4a, ogg
    
    **Returns:**
    ```json
    {
      "text": "Your transcribed text here"
    }
    ```
    
    **Note:** First run will download ~150MB Whisper model. Subsequent runs use cached model.
    """
    model = await get_whisper_model()
    content = await audio.read()
    # Write to temp file (Whisper needs file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        segments, _ = await asyncio.to_thread(model.transcribe, tmp_path)
        text = " ".join([seg.text for seg in segments]).strip()
    finally:
        os.unlink(tmp_path)
    return {"text": text}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
