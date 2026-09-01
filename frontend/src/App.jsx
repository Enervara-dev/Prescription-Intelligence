import { useState, useCallback } from 'react'
import axios from 'axios'
import { UploadCloud, FileText, Pill, User, Clock, CheckCircle, AlertCircle, Loader, RefreshCw } from 'lucide-react'
import './App.css'

const API_URL = 'http://localhost:8000'

function ConfidenceBadge({ value }) {
  const color = value >= 85 ? 'badge-green' : value >= 65 ? 'badge-yellow' : 'badge-red'
  return <span className={`badge ${color}`}>{value}%</span>
}

function MedicineTable({ medicines }) {
  if (!medicines || medicines.length === 0) {
    return (
      <div className="empty-state">
        <AlertCircle className="empty-icon" />
        <p>No medicines detected. Try a clearer image or ensure lighting is bright.</p>
      </div>
    )
  }

  return (
    <div className="table-wrapper">
      <table className="med-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Medicine Name</th>
            <th>Dosage</th>
            <th>Frequency</th>
            <th>Duration</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {medicines.map((med, idx) => (
            <tr key={idx}>
              <td className="num-cell">{idx + 1}</td>
              <td className="name-cell">
                <Pill className="inline-icon" />
                {med.name}
              </td>
              <td>{med.dosage}</td>
              <td>{med.frequency}</td>
              <td>{med.duration}</td>
              <td><ConfidenceBadge value={med.confidence} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function UploadZone({ onFileSelect, file, onProcess, loading }) {
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) onFileSelect(dropped)
  }, [onFileSelect])

  return (
    <div className="upload-zone">
      <div
        className={`drop-area ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*,application/pdf"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files[0] && onFileSelect(e.target.files[0])}
        />
        {file ? (
          <>
            <FileText className="upload-icon active" />
            <p className="file-name">{file.name}</p>
            <p className="file-hint">Click or drag to choose another file</p>
          </>
        ) : (
          <>
            <UploadCloud className="upload-icon" />
            <p className="upload-label">Drop prescription here or click to browse</p>
            <p className="upload-hint">Supports JPG, PNG, PDF (Handwritten & Printed)</p>
          </>
        )}
      </div>

      <button
        className={`process-btn ${loading ? 'loading' : ''}`}
        onClick={onProcess}
        disabled={!file || loading}
      >
        {loading ? (
          <><Loader className="spin" /> Processing OCR & Auto-Orienting...</>
        ) : (
          <><CheckCircle /> Extract Medicines</>
        )}
      </button>
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileSelect = (f) => {
    setFile(f)
    setResult(null)
    setError(null)
    if (f.type.startsWith('image/')) {
      setPreview(URL.createObjectURL(f))
    } else {
      setPreview(null)
    }
  }

  const handleProcess = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await axios.post(`${API_URL}/api/process`, form)
      setResult(res.data)
      if (res.data.image_preview) {
        setPreview(res.data.image_preview)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Processing failed')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <Pill className="logo-icon" />
            <span className="logo-text">RxReader</span>
            <span className="logo-badge">AI</span>
          </div>
          <p className="tagline">Prescription OCR — Handwritten & Printed</p>
        </div>
      </header>

      <main className="app-main">
        {!result ? (
          /* Upload Screen */
          <div className="upload-screen">
            <div className="upload-card">
              <h2 className="card-title">Upload Prescription</h2>
              <p className="card-sub">Upload any handwritten or printed prescription to extract medicine details instantly.</p>
              <UploadZone
                onFileSelect={handleFileSelect}
                file={file}
                onProcess={handleProcess}
                loading={loading}
              />
              {error && (
                <div className="error-box">
                  <AlertCircle className="error-icon" />
                  <span>{error}</span>
                </div>
              )}
            </div>

            {/* Feature pills */}
            <div className="features">
              {['Auto-Orientation', 'EasyOCR Dual-Pass', 'RapidFuzz Matching', 'PDF & Image Preview'].map(f => (
                <span key={f} className="feature-pill">{f}</span>
              ))}
            </div>
          </div>
        ) : (
          /* Results Screen */
          <div className="results-screen">
            {/* Left: Image Preview */}
            <div className="preview-panel">
              <div className="panel-header">
                <FileText className="panel-icon" />
                <span>Prescription Document</span>
              </div>
              <div className="image-box">
                {preview ? (
                  <img src={preview} alt="Prescription Preview" className="prescription-img" />
                ) : (
                  <div className="pdf-placeholder">
                    <FileText className="pdf-icon" />
                    <p>{file?.name}</p>
                    <span>Prescription Loaded</span>
                  </div>
                )}
              </div>
              <button className="reset-btn" onClick={handleReset}>
                <RefreshCw size={14} style={{ display: 'inline', marginRight: 6 }} /> Process Another
              </button>
            </div>

            {/* Right: Results */}
            <div className="results-panel">
              {/* Raw Extracted Text */}
              <div className="info-card raw-card">
                <div className="info-header">
                  <FileText className="info-icon" />
                  <span>Raw Extracted Text</span>
                </div>
                <div className="raw-text-box">
                  <pre className="raw-text">{result.debug?.full_text || 'No text extracted.'}</pre>
                </div>
              </div>

              {/* Patient Info */}
              <div className="info-card patient-card">
                <div className="info-header">
                  <User className="info-icon" />
                  <span>Patient & Doctor Information</span>
                </div>
                <p className="patient-text">
                  {result.patient_info || 'Patient details not detected clearly.'}
                </p>
              </div>

              {/* Medicines */}
              <div className="info-card meds-card">
                <div className="info-header">
                  <Pill className="info-icon" />
                  <span>Medicines Detected ({result.medicines?.length || 0})</span>
                </div>
                <MedicineTable medicines={result.medicines} />
              </div>

              {/* Stats */}
              <div className="stats-row">
                <div className="stat">
                  <Clock className="stat-icon" />
                  <span>{result.processing_time_sec}s processing time</span>
                </div>
                <div className="stat">
                  <FileText className="stat-icon" />
                  <span>{result.debug?.word_count} words extracted</span>
                </div>
                <div className="stat">
                  <CheckCircle className="stat-icon" />
                  <span>{result.medicines?.length} medicines found</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
