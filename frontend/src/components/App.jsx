import React, { useEffect, useRef, useState } from 'react'

const BACKEND_URL = 'http://localhost:8000'

function Badge({ score }) {
  let color = 'gray'
  if (score < 0.4) color = 'green'
  else if (score < 0.7) color = 'orange'
  else color = 'red'
  return (
    <span style={{
      backgroundColor: color,
      color: 'white',
      padding: '4px 8px',
      borderRadius: '12px',
      marginLeft: '8px'
    }}>
      {score.toFixed(2)}
    </span>
  )
}

function LineChart({ data }) {
  // Simple inline SVG line chart with fixed 0-1 Y axis
  const width = 600
  const height = 200
  const padding = 30
  if (!data.length) {
    return (
      <svg width={width} height={height} style={{ border: '1px solid #ccc', marginTop: 8 }}>
        <text x={width / 2} y={height / 2} textAnchor="middle" fill="#999">No data yet</text>
      </svg>
    )
  }
  const minY = 0
  const maxY = 1
  const points = data.map((d, i) => {
    const x = padding + (i / Math.max(data.length - 1, 1)) * (width - 2 * padding)
    const y = padding + (1 - (d.score - minY) / (maxY - minY)) * (height - 2 * padding)
    return `${x},${y}`
  }).join(' ')
  // Y axis labels
  const yLabels = [0, 0.25, 0.5, 0.75, 1]
  return (
    <svg width={width} height={height} style={{ border: '1px solid #ccc', marginTop: 8 }}>
      {/* Y axis */}
      <line x1={padding} x2={padding} y1={padding} y2={height - padding} stroke="#aaa" />
      {/* X axis */}
      <line x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} stroke="#aaa" />
      {/* Y labels */}
      {yLabels.map(v => {
        const y = padding + (1 - v) * (height - 2 * padding)
        return (
          <g key={v}>
            <line x1={padding - 4} x2={padding} y1={y} y2={y} stroke="#aaa" />
            <text x={padding - 8} y={y + 4} textAnchor="end" fontSize={10} fill="#666">{v}</text>
          </g>
        )
      })}
      {/* Data line */}
      <polyline fill="none" stroke="#1976d2" strokeWidth="2" points={points} />
      {/* Data points */}
      {data.map((d, i) => {
        const x = padding + (i / Math.max(data.length - 1, 1)) * (width - 2 * padding)
        const y = padding + (1 - (d.score - minY) / (maxY - minY)) * (height - 2 * padding)
        return <circle key={i} cx={x} cy={y} r={4} fill="#1976d2" />
      })}
    </svg>
  )
}

export default function App() {
  const [list, setList] = useState([]) // {text, score, timestamp}
  const [transcript, setTranscript] = useState('')
  const [interim, setInterim] = useState('')
  const [recording, setRecording] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const [latest, setLatest] = useState(null) // {score, flag, explanation}
  const [manualText, setManualText] = useState('')
  const [manualLoading, setManualLoading] = useState(false)
  const [useWhisper, setUseWhisper] = useState(true) // Toggle: true=Whisper, false=WebSpeech
  const [whisperRecording, setWhisperRecording] = useState(false)
  const [whisperProcessing, setWhisperProcessing] = useState(false)
  const recognitionRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setErrorMsg('Web Speech API not supported in this browser. Please use Chrome on desktop.')
      return
    }
    const recog = new SpeechRecognition()
    recog.continuous = true
    recog.interimResults = true
    recog.lang = 'en-US'
    recog.onresult = async (event) => {
      let finalChunk = ''
      let interimText = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i]
        const text = res[0].transcript
        if (res.isFinal) {
          finalChunk += text + ' '
        } else {
          interimText = text
        }
      }
      // Show interim live (doesn't get sent to backend)
      setInterim(interimText.trim())

      finalChunk = finalChunk.trim()
      if (finalChunk) {
        setTranscript(prev => (prev ? prev + '\n' + finalChunk : finalChunk))
        const ts = Date.now()
        try {
          const resp = await fetch(`${BACKEND_URL}/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: finalChunk })
          })
          const json = await resp.json()
          setLatest({ score: json.score ?? 0, flag: !!json.flag, explanation: json.explanation || '' })
          setList(prev => [...prev, { text: finalChunk, score: json.score ?? 0, timestamp: ts }])
        } catch (e) {
          console.error(e)
          setErrorMsg('Failed to reach backend /detect. Ensure backend is running on localhost:8000.')
        }
      }
    }
    recog.onerror = (e) => console.error('recognition error', e)
    recog.onstart = () => setErrorMsg('')
    recognitionRef.current = recog
  }, [])

  const start = async () => {
    if (!recognitionRef.current || recording) return
    try {
      // Must be triggered by user gesture; browsers may block if not
      recognitionRef.current.start()
      setRecording(true)
      setErrorMsg('')
    } catch (e) {
      console.error(e)
      setErrorMsg('Microphone start failed. Check site permissions and use Chrome on localhost over http.')
    }
  }
  const stop = () => {
    if (recognitionRef.current && recording) {
      recognitionRef.current.stop()
      setRecording(false)
    }
  }

  // Whisper-based recording (MediaRecorder → /stt → /detect)
  const startWhisper = async () => {
    if (whisperRecording) return
    setErrorMsg('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      audioChunksRef.current = []
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        await processWhisperAudio(blob)
      }
      mediaRecorder.start()
      mediaRecorderRef.current = mediaRecorder
      setWhisperRecording(true)
    } catch (e) {
      console.error(e)
      setErrorMsg('Microphone access denied or unavailable.')
    }
  }

  const stopWhisper = () => {
    if (mediaRecorderRef.current && whisperRecording) {
      mediaRecorderRef.current.stop()
      setWhisperRecording(false)
    }
  }

  const processWhisperAudio = async (blob) => {
    setWhisperProcessing(true)
    setInterim('[Processing audio with Whisper...]')
    try {
      // 1. Send audio to /stt for transcription
      const formData = new FormData()
      formData.append('audio', blob, 'recording.webm')
      const sttResp = await fetch(`${BACKEND_URL}/stt`, { method: 'POST', body: formData })
      const sttJson = await sttResp.json()
      const text = sttJson.text || ''
      setInterim('')
      if (!text.trim()) {
        setErrorMsg('No speech detected in audio.')
        setWhisperProcessing(false)
        return
      }
      setTranscript(prev => (prev ? prev + '\n' + text : text))
      // 2. Send transcript to /detect
      const ts = Date.now()
      const detectResp = await fetch(`${BACKEND_URL}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })
      const detectJson = await detectResp.json()
      setLatest({ score: detectJson.score ?? 0, flag: !!detectJson.flag, explanation: detectJson.explanation || '' })
      setList(prev => [...prev, { text, score: detectJson.score ?? 0, timestamp: ts }])
    } catch (e) {
      console.error(e)
      setErrorMsg('Failed to process audio. Ensure backend is running.')
    } finally {
      setWhisperProcessing(false)
    }
  }

  // Manual text detection (fallback when mic unavailable)
  const detectManual = async () => {
    if (!manualText.trim()) return
    setManualLoading(true)
    setErrorMsg('')
    const ts = Date.now()
    try {
      const resp = await fetch(`${BACKEND_URL}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: manualText.trim() })
      })
      const json = await resp.json()
      setLatest({ score: json.score ?? 0, flag: !!json.flag, explanation: json.explanation || '' })
      setList(prev => [...prev, { text: manualText.trim(), score: json.score ?? 0, timestamp: ts }])
      setTranscript(prev => (prev ? prev + '\n' + manualText.trim() : manualText.trim()))
      setManualText('')
    } catch (e) {
      console.error(e)
      setErrorMsg('Failed to reach backend /detect.')
    } finally {
      setManualLoading(false)
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui' }}>
      <h2>Real-Time AI-Generated Answer Detection (POC)</h2>

      {/* Mode toggle */}
      <div style={{ marginBottom: 12 }}>
        <label style={{ marginRight: 16 }}>
          <input
            type="radio"
            checked={useWhisper}
            onChange={() => setUseWhisper(true)}
          /> Whisper STT (better accuracy)
        </label>
        <label>
          <input
            type="radio"
            checked={!useWhisper}
            onChange={() => setUseWhisper(false)}
          /> Web Speech API (real-time, Chrome only)
        </label>
      </div>

      {/* Recording controls */}
      <div style={{ marginBottom: 12 }}>
        {useWhisper ? (
          <>
            <button onClick={startWhisper} disabled={whisperRecording || whisperProcessing} style={{ marginRight: 8 }}>
              {whisperRecording ? '🔴 Recording...' : 'Start Recording (Whisper)'}
            </button>
            <button onClick={stopWhisper} disabled={!whisperRecording}>
              Stop & Transcribe
            </button>
            {whisperProcessing && <span style={{ marginLeft: 8, color: '#666' }}>Processing...</span>}
          </>
        ) : (
          <>
            <button onClick={start} disabled={recording} style={{ marginRight: 8 }}>Start Recording</button>
            <button onClick={stop} disabled={!recording}>Stop Recording</button>
          </>
        )}
      </div>

      <div style={{ marginBottom: 12 }}>
        <h4>Or type/paste text to analyze:</h4>
        <textarea
          value={manualText}
          onChange={e => setManualText(e.target.value)}
          placeholder="Type or paste answer text here..."
          rows={3}
          style={{ width: '100%', maxWidth: 600, padding: 8 }}
        />
        <br />
        <button onClick={detectManual} disabled={manualLoading || !manualText.trim()} style={{ marginTop: 4 }}>
          {manualLoading ? 'Analyzing...' : 'Analyze Text'}
        </button>
      </div>

      <div>
        <h3>Live Transcript</h3>
        <pre style={{ border: '1px solid #ccc', padding: 12, minHeight: 120, whiteSpace: 'pre-wrap' }}>
          {interim ? `[interim] ${interim}\n` : ''}{transcript}
        </pre>
      </div>

      <div>
        <h3>Scores</h3>
        {latest && (
          <div style={{ marginBottom: 8 }}>
            <strong>Latest Detection:</strong>
            <Badge score={latest.score} />
            <span style={{ marginLeft: 8 }}>
              {latest.flag ? 'AI-like detected' : 'Human-like'} — {latest.explanation}
            </span>
          </div>
        )}
        {list.map((item, idx) => (
          <div key={idx} style={{ display: 'flex', alignItems: 'center', marginBottom: 6 }}>
            <div style={{ flex: 1 }}>{item.text}</div>
            <Badge score={item.score} />
          </div>
        ))}
      </div>

      <div>
        <h3>Score History</h3>
        <LineChart data={list} />
      </div>

      {errorMsg && (
        <div style={{ marginTop: 12, color: 'crimson' }}>
          {errorMsg}
        </div>
      )}
    </div>
  )
}
