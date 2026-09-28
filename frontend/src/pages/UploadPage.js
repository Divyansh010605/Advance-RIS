import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePrediction } from '../hooks/usePrediction';
import { parseDicom } from '../utils/dicom';

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'application/dicom', '']; // empty string for cases where browser doesn't know mimetype
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.dcm'];
const MAX_SIZE_MB = 20; // DICOM files are often larger, up to 20MB

export default function UploadPage() {
  const navigate = useNavigate();
  const { predict, loading, error } = usePrediction();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isProcessingDicom, setIsProcessingDicom] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  async function handleFile(selected) {
    if (!selected) return;
    
    const isDicom = selected.name.toLowerCase().endsWith('.dcm');
    const hasValidExtension = ALLOWED_EXTENSIONS.some(ext => selected.name.toLowerCase().endsWith(ext));
    
    if (!hasValidExtension && !ALLOWED_TYPES.includes(selected.type)) {
      alert('Unsupported format. Please use JPEG, PNG, WebP, BMP, or DCM.');
      return;
    }
    if (selected.size > MAX_SIZE_MB * 1024 * 1024) {
      alert(`File too large. Maximum size is ${MAX_SIZE_MB} MB.`);
      return;
    }
    
    if (isDicom) {
      setIsProcessingDicom(true);
      try {
        const { file: jpgFile, previewUrl } = await parseDicom(selected);
        setFile(jpgFile);
        setPreview(previewUrl);
      } catch (err) {
        alert(err.message);
      } finally {
        setIsProcessingDicom(false);
      }
    } else {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
    }
  }

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    handleFile(dropped);
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    try {
      const result = await predict(file);
      navigate('/results', { state: { result, previewUrl: preview } });
    } catch {
      // error displayed from hook
    }
  }

  return (
    <main className="upload-page container" role="main">
      <header className="page-header">
        <h1>Upload Chest X-Ray</h1>
        <p className="text-muted">
          Upload a PA (posteroanterior) view chest X-ray. The AI model will screen for 14 major lung diseases and generate explainability heatmaps.
        </p>
      </header>

      {error && (
        <div className="alert alert-error mb-4" role="alert" id="upload-error">
          Error: {error}
        </div>
      )}

      <div className="grid-2">
        {/* Drop Zone */}
        <div className="card">
          <form onSubmit={handleSubmit} id="upload-form">
            <div
              id="drop-zone"
              className={`drop-zone${dragOver ? ' drag-over' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
              aria-label="Drop zone for chest X-ray image"
              onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
            >
              <div className="drop-zone__icon">File</div>
              {file ? (
                <p className="drop-zone__text">
                  <strong>{file.name}</strong>
                  <br />
                  <span className="text-muted text-sm">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                </p>
              ) : (
                <p className="drop-zone__text">
                  <strong>Click to browse</strong> or drag & drop<br />
                  <span className="text-muted text-sm">JPEG, PNG, WebP, BMP, DCM · Max {MAX_SIZE_MB} MB</span>
                </p>
              )}
              {preview && (
                <img
                  id="image-preview"
                  src={preview}
                  alt="Uploaded chest X-ray preview"
                  className="preview-img"
                />
              )}
            </div>

            <input
              ref={inputRef}
              id="file-input"
              type="file"
              accept={[...ALLOWED_TYPES, ...ALLOWED_EXTENSIONS].join(',')}
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files[0])}
              aria-label="File input for chest X-ray"
            />

            <button
              id="analyze-btn"
              type="submit"
              className="btn btn-primary btn-lg mt-4"
              style={{ width: '100%', justifyContent: 'center' }}
              disabled={!file || loading || isProcessingDicom}
            >
              {isProcessingDicom ? 'Processing DICOM...' : loading ? 'Analyzing…' : 'Analyze X-Ray'}
            </button>
          </form>
        </div>

        {/* Instructions */}
        <div className="card card-sm" style={{ alignSelf: 'start' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 16 }}>Instructions</h2>
          <ol style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[
              ['1', 'Upload a PA chest X-ray image'],
              ['2', 'Supported: JPEG, PNG, WebP, BMP, DCM'],
              ['3', `Maximum file size: ${MAX_SIZE_MB} MB`],
              ['4', 'Click "Analyze X-Ray" to run the AI model'],
              ['5', 'View prediction score and Grad-CAM explainability heatmap'],
            ].map(([num, text]) => (
              <li key={num} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <span style={{ background: 'var(--color-primary)', color: '#0f172a', borderRadius: '50%', width: 24, height: 24, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontWeight: 700, fontSize: '0.8rem' }}>{num}</span>
                <span className="text-muted text-sm">{text}</span>
              </li>
            ))}
          </ol>

          <div className="divider" />

          <div className="alert alert-warning text-sm" role="note">
            Note: This tool is for research purposes only and should not replace professional medical diagnosis.
          </div>
        </div>
      </div>

      {loading && (
        <div className="spinner-overlay" role="status" aria-live="polite">
          <div className="spinner" />
          <p className="text-muted">Running AI inference and generating Grad-CAM…</p>
        </div>
      )}
    </main>
  );
}
