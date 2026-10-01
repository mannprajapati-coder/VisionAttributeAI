// VisionAttributeAI - Unified Multi-Person Recognition Pipeline Controller (Image, Video, Live Camera)

let currentSelectedFile = null;
let currentSelectedVideoFile = null;
let currentParSelectedFile = null;

// Palette for multiple persons (Blue, Violet, Mint, Amber, Coral, Cyan, Pink, Lavender)
const PERSON_COLORS = [
    '#4F8CFF', // Blue (Person 01)
    '#8B5CF6', // Violet (Person 02)
    '#22C7A5', // Mint (Person 03)
    '#F5B942', // Amber (Person 04)
    '#FF5C6C', // Coral/Red (Person 05)
    '#06B6D4', // Cyan (Person 06)
    '#EC4899', // Pink (Person 07)
    '#A78BFA'  // Lavender (Person 08)
];

function getPersonColor(personId) {
    const idx = Math.max(0, (personId - 1) % PERSON_COLORS.length);
    return PERSON_COLORS[idx];
}

const COLOR_HEX_MAP = {
    'black': '#18181b',
    'white': '#f8fafc',
    'grey': '#94a3b8',
    'gray': '#94a3b8',
    'red': '#ef4444',
    'blue': '#3b82f6',
    'green': '#22c55e',
    'yellow': '#eab308',
    'orange': '#f97316',
    'purple': '#a855f7',
    'brown': '#78350f',
    'pink': '#ec4899',
    'beige': '#d4d4d8',
    'navy': '#1e3a8a',
    'khaki': '#a3a375'
};

function getColorHex(colorName) {
    if (!colorName) return '#94a3b8';
    return COLOR_HEX_MAP[colorName.toLowerCase().trim()] || '#94a3b8';
}

function formatClothing(color, type) {
    if (!type || type === 'Not Visible' || type === 'Insufficient Evidence') return type || 'Unknown';
    if (!color || color === 'Unknown' || color === 'Not Visible' || color === 'Insufficient Evidence') return type;
    const hex = getColorHex(color);
    return `<span class="color-swatch-pill"><span class="color-swatch-dot" style="background-color: ${hex};"></span>${color}</span> <span>${type}</span>`;
}

function formatClothingText(color, type) {
    if (!type || type === 'Not Visible' || type === 'Insufficient Evidence') return type || 'Unknown';
    if (!color || color === 'Unknown' || color === 'Not Visible' || color === 'Insufficient Evidence') return type;
    return `${color} ${type}`;
}

function formatStatusText(val) {
    if (!val) return 'Not Detected';
    const v = val.toString().trim();
    if (v === 'Detected' || v === 'WatchDetected' || v === 'Watch Detected') return 'Detected';
    if (v === 'NotDetected' || v === 'Not Detected' || v === 'NoWatchDetected' || v === 'No Watch' || v === 'No Watch Detected' || v === 'Unknown') return 'Not Detected';
    if (v === 'InsufficientVisualEvidence' || v === 'Insufficient Evidence') return 'Insufficient Evidence';
    if (v === 'NotVisible' || v === 'Not Visible') return 'Not Visible';
    if (v === 'ModelUnavailable' || v === 'Model Unavailable') return 'Model Unavailable';
    return v;
}

function formatBrandBadge(state) {
    if (!state) return '';
    const s = state.toLowerCase();
    if (s === 'brandstable' || s === 'stable') return '<span class="state-badge state-stable">BrandStable</span>';
    if (s === 'brandcandidate' || s === 'candidate') return '<span class="state-badge state-analyzing">BrandCandidate</span>';
    if (s === 'brandunknown' || s === 'unknown') return '<span class="state-badge state-notvisible">BrandUnknown</span>';
    return `<span class="state-badge state-${s}">${state}</span>`;
}

function isNegativeBrand(name) {
    if (!name) return true;
    const n = name.toString().trim().toLowerCase();
    return n === 'not visible' ||
           n === 'not detected' ||
           n === 'no logo candidate' ||
           n === 'model unavailable' ||
           n === 'modelunavailable' ||
           n === 'unsupported region' ||
           n === 'insufficient visual evidence' ||
           n === 'insufficient evidence' ||
           n === 'brandunknown' ||
           n === 'unknown' ||
           n === 'none';
}

function formatBrandDisplay(p) {
    if (!p) return 'BrandUnknown';
    if (p.brands && p.brands.hasAnyAcceptedBrand) {
        const parts = [];
        if (p.brands.upperBrand && !isNegativeBrand(p.brands.upperBrand.brandName)) {
            parts.push(`Upper: ${p.brands.upperBrand.brandName} (${(p.brands.upperBrand.similarity * 100).toFixed(0)}%)`);
        }
        if (p.brands.lowerBrand && !isNegativeBrand(p.brands.lowerBrand.brandName)) {
            parts.push(`Lower: ${p.brands.lowerBrand.brandName} (${(p.brands.lowerBrand.similarity * 100).toFixed(0)}%)`);
        }
        if (p.brands.watchBrand && !isNegativeBrand(p.brands.watchBrand.brandName)) {
            parts.push(`Watch: ${p.brands.watchBrand.brandName} (${(p.brands.watchBrand.similarity * 100).toFixed(0)}%)`);
        }
        if (p.brands.shoeBrand && !isNegativeBrand(p.brands.shoeBrand.brandName)) {
            parts.push(`Shoes: ${p.brands.shoeBrand.brandName} (${(p.brands.shoeBrand.similarity * 100).toFixed(0)}%)`);
        }
        if (p.brands.bagBrand && !isNegativeBrand(p.brands.bagBrand.brandName)) {
            parts.push(`Bag: ${p.brands.bagBrand.brandName} (${(p.brands.bagBrand.similarity * 100).toFixed(0)}%)`);
        }
        if (parts.length > 0) return parts.join(' • ');
        if (p.brands.topDetectedBrand && !isNegativeBrand(p.brands.topDetectedBrand)) {
            return `${p.brands.topDetectedBrand} (${(p.brands.topBrandSimilarity * 100).toFixed(0)}%)`;
        }
    }
    if (p.brandName && !isNegativeBrand(p.brandName)) {
        return `${p.brandName} ${p.brandConfidence > 0 ? `(${(p.brandConfidence * 100).toFixed(0)}%)` : ''}`;
    }
    return 'BrandUnknown';
}

function formatStateBadge(state, isFullyStable) {
    if (isFullyStable) return '<span class="state-badge state-stable">Stable</span>';
    if (!state) return '<span class="state-badge state-analyzing">Analyzing</span>';
    const s = state.toString().toLowerCase().replace(/_/g, '');
    if (s.includes('stable')) return '<span class="state-badge state-stable">Stable</span>';
    if (s.includes('insufficient')) return '<span class="state-badge state-insufficient">Insufficient Evidence</span>';
    if (s.includes('notvisible')) return '<span class="state-badge state-notvisible">Not Visible</span>';
    if (s.includes('analyzing') || s.includes('learning')) return '<span class="state-badge state-analyzing">Analyzing</span>';
    return `<span class="state-badge state-analyzing">${state}</span>`;
}

function renderRegionalBrandPanel(p) {
    if (!p) return '';
    const b = p.brands || {};
    const detected = [];

    const checkAndAdd = (region, brandObj) => {
        if (!brandObj) return;
        const name = brandObj.brandName;
        if (name && !isNegativeBrand(name)) {
            const sim = brandObj.similarity ? `${(brandObj.similarity * 100).toFixed(0)}%` : '';
            detected.push({ region, name, sim });
        }
    };

    checkAndAdd('Upper', b.upperBrand);
    checkAndAdd('Lower', b.lowerBrand);
    checkAndAdd('Watch', b.watchBrand);
    checkAndAdd('Shoes', b.shoeBrand);
    checkAndAdd('Bag', b.bagBrand);

    if (detected.length === 0 && p.brandName && !isNegativeBrand(p.brandName)) {
        const sim = p.brandConfidence > 0 ? `${(p.brandConfidence * 100).toFixed(0)}%` : '';
        detected.push({ region: 'Garment', name: p.brandName, sim });
    }

    let contentHtml = '';
    if (detected.length > 0) {
        contentHtml = `
            <div class="v-brand-chips-grid">
                ${detected.map(d => `
                    <span class="v-brand-chip highlight" title="${d.region}: ${d.name} ${d.sim ? `(${d.sim})` : ''}">
                        <span class="v-brand-chip-region">${d.region}:</span>
                        <span>${d.name}</span>
                        ${d.sim ? `<span style="opacity: 0.8; font-size: 0.68rem;">(${d.sim})</span>` : ''}
                    </span>
                `).join('')}
            </div>
        `;
    } else {
        contentHtml = `<span class="v-brand-empty">No visible brand signatures detected</span>`;
    }

    return `
        <div class="v-brand-panel">
            <div class="v-brand-panel-header">
                <span class="v-brand-panel-title">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="color:var(--accent-violet);"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                    Brand Intelligence
                </span>
                ${detected.length > 0 ? `<span class="state-badge state-stable">${detected.length} Detected</span>` : '<span class="state-badge state-insufficient">None</span>'}
            </div>
            ${contentHtml}
        </div>
    `;
}

function renderRegionalBrandRows(p) {
    return renderRegionalBrandPanel(p);
}

document.addEventListener('DOMContentLoaded', () => {
    setupDropzone();
    setupVideoDropzone();
    setupParDropzone();
    checkHealth();
    populateCameraDevices();
    setupLiveCanvasClick();
});

function setupLiveCanvasClick() {
    const canvas = document.getElementById('cameraOverlayCanvas');
    if (!canvas) return;

    canvas.addEventListener('click', (e) => {
        if (!currentLivePersonsList || currentLivePersonsList.length === 0) return;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        // Find clicked person
        const clicked = currentLivePersonsList.find(p => {
            const b = p.boundingBox;
            return clickX >= b.x && clickX <= (b.x + b.width) &&
                   clickY >= b.y && clickY <= (b.y + b.height);
        });

        if (clicked) {
            selectLivePerson(clicked.personId);
        }
    });
}

// -------------------------------------------------------------
// Tab Switching & Developer Diagnostics Collapsible
// -------------------------------------------------------------

function switchTab(mode) {
    const tabUpload = document.getElementById('tabUpload');
    const tabVideo = document.getElementById('tabVideo');
    const tabCamera = document.getElementById('tabCamera');
    const panelUpload = document.getElementById('panelUpload');
    const panelVideo = document.getElementById('panelVideo');
    const panelCamera = document.getElementById('panelCamera');

    hideNotification();

    [tabUpload, tabVideo, tabCamera].forEach(tab => tab?.classList.remove('active'));
    [panelUpload, panelVideo, panelCamera].forEach(panel => panel?.classList.remove('active'));

    if (mode === 'upload') {
        tabUpload?.classList.add('active');
        panelUpload?.classList.add('active');
    } else if (mode === 'video') {
        tabVideo?.classList.add('active');
        panelVideo?.classList.add('active');
    } else {
        tabCamera?.classList.add('active');
        panelCamera?.classList.add('active');
    }
}

function toggleDevDiagnostics() {
    const content = document.getElementById('devDiagnosticsContent');
    const arrow = document.getElementById('devArrow');
    if (!content) return;

    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
        if (arrow) arrow.textContent = '▼ Click to collapse Legacy PAR Test';
    } else {
        content.style.display = 'none';
        if (arrow) arrow.textContent = '▶ Click to expand Legacy PAR Test';
    }
}

// -------------------------------------------------------------
// 1. TAB 1 — ANALYZE IMAGE (Multi-Person Single Image Workflow)
// -------------------------------------------------------------

function setupDropzone() {
    const dropzone = document.getElementById('dropzone');
    if (!dropzone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = 'var(--primary-color)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = 'var(--border-color)';
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    }, false);
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleSelectedFile(files[0]);
    }
}

function handleSelectedFile(file) {
    hideNotification();

    const validExtensions = ['.jpg', '.jpeg', '.jfif', '.png'];
    const fileName = file.name.toLowerCase();
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));

    if (!hasValidExtension) {
        showNotification('Invalid file format. Only JPG, JPEG, JFIF, and PNG images are allowed.', 'error');
        return;
    }

    const maxSizeBytes = 10 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
        showNotification(`File is too large (${formatFileSize(file.size)}). Maximum allowed size is 10 MB.`, 'error');
        return;
    }

    currentSelectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        const previewArea = document.getElementById('previewArea');
        const dropzone = document.getElementById('dropzone');
        const imagePreview = document.getElementById('imagePreview');
        const previewFilename = document.getElementById('previewFilename');
        const previewDimensions = document.getElementById('previewDimensions');
        const pipelineResultsArea = document.getElementById('pipelineResultsArea');

        imagePreview.onload = () => {
            previewDimensions.textContent = `${imagePreview.naturalWidth} × ${imagePreview.naturalHeight} px`;
            const canvas = document.getElementById('overlayCanvas');
            if (canvas) {
                canvas.width = imagePreview.naturalWidth;
                canvas.height = imagePreview.naturalHeight;
            }
        };

        imagePreview.src = e.target.result;
        imagePreview.style.cursor = 'pointer';
        imagePreview.title = 'Click to preview full image';
        imagePreview.onclick = () => openImageModal(imagePreview.src, 'Uploaded Full Image', `${file.name} (${imagePreview.naturalWidth || ''} × ${imagePreview.naturalHeight || ''} px)`);
        previewFilename.textContent = `${file.name} (${formatFileSize(file.size)})`;
        pipelineResultsArea.style.display = 'none';

        dropzone.style.display = 'none';
        previewArea.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

async function onAnalyzeClicked() {
    if (!currentSelectedFile) {
        showNotification('Please select an image file first.', 'error');
        return;
    }

    const btn = document.getElementById('btnAnalyzeImage');
    const btnText = document.getElementById('btnAnalyzeText');
    const pipelineResultsArea = document.getElementById('pipelineResultsArea');

    try {
        btn.disabled = true;
        btnText.textContent = 'Analyzing All People...';
        hideNotification();
        clearOverlayCanvas();

        const formData = new FormData();
        formData.append('file', currentSelectedFile);

        const response = await fetch('/api/analysis/image', {
            method: 'POST',
            body: formData
        });

        let data;
        try {
            data = await response.json();
        } catch (jsonErr) {
            if (!response.ok) {
                showNotification(`Server error (${response.status}: ${response.statusText || 'Service Unavailable'}). Please ensure the backend is running.`, 'error');
            } else {
                showNotification('Invalid response received from server.', 'error');
            }
            pipelineResultsArea.style.display = 'none';
            return;
        }

        if (!response.ok || !data.success) {
            const errorMsg = data.errorMessage || `Server returned error (${response.status}: ${response.statusText})`;
            showNotification(errorMsg, 'error');
            pipelineResultsArea.style.display = 'none';
            return;
        }

        document.getElementById('metricResolution').textContent = `${data.imageWidth} × ${data.imageHeight} px`;
        document.getElementById('metricElapsed').textContent = `${data.totalElapsedMs.toFixed(1)} ms`;
        document.getElementById('metricDetectionMs').textContent = `${data.detectionMs.toFixed(1)} ms`;
        document.getElementById('metricPersons').textContent = (data.persons ? data.persons.length : 0).toString();

        renderImageDetections(data.persons, data.imageWidth, data.imageHeight);

        pipelineResultsArea.style.display = 'block';
        showNotification(`Analysis complete. Found and evaluated ${data.persons ? data.persons.length : 0} person(s).`, 'success');
    } catch (err) {
        console.error('Pipeline processing error:', err);
        showNotification(`Error during image analysis: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Analyze All People';
    }
}

function renderImageDetections(persons, origWidth, origHeight) {
    const list = document.getElementById('detectionsList');
    list.innerHTML = '';

    const canvas = document.getElementById('overlayCanvas');
    const img = document.getElementById('imagePreview');

    if (!img.naturalWidth || !img.naturalHeight) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!persons || persons.length === 0) {
        list.innerHTML = '<div class="no-detections-message">No persons detected above confidence threshold.</div>';
        return;
    }

    persons.forEach((p) => {
        const box = p.boundingBox;
        const color = getPersonColor(p.personId);
        const confPercent = (p.detectionConfidence * 100).toFixed(1);

        // 1. Draw Bounding Box on Canvas
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(3, Math.round(canvas.width / 300));
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        // 2. Draw Label Header
        const labelText = `Person #${p.personId} (${confPercent}%)`;
        const fontSize = Math.max(14, Math.round(canvas.width / 50));
        ctx.font = `600 ${fontSize}px Inter, sans-serif`;

        const textMetrics = ctx.measureText(labelText);
        const padding = 6;
        const labelHeight = fontSize + padding * 2;
        const labelWidth = textMetrics.width + padding * 2;

        let labelY = box.y - labelHeight;
        if (labelY < 0) labelY = box.y;

        ctx.fillStyle = color;
        ctx.fillRect(box.x, labelY, labelWidth, labelHeight);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, box.x + padding, labelY + fontSize + padding / 2);

        // 3. Render Independent Person Card
        const card = document.createElement('div');
        card.className = 'detection-card';
        card.style.borderTop = `4px solid ${color}`;
        const personIdFormatted = String(p.personId).padStart(2, '0');

        card.innerHTML = `
            <div class="detection-card-header">
                <div class="person-tag-wrap">
                    <span class="person-color-dot" style="background-color: ${color}; color: ${color};"></span>
                    <span class="detection-label" style="color: ${color};">PERSON ${personIdFormatted}</span>
                    <span class="detection-conf-badge">${confPercent}% Conf</span>
                </div>
                <span class="state-badge state-stable">Single Image</span>
            </div>
            
            <div class="v-person-body">
                ${p.cropDataUrl ? `
                <div class="v-person-crop" title="Click to enlarge person crop" onclick="openImageModal('${p.cropDataUrl}', 'Person #${p.personId} Crop', '${p.appearanceSex} • ${formatClothingText(p.upperColor, p.upperType)} • ${formatClothingText(p.lowerColor, p.lowerType)}')">
                    <img src="${p.cropDataUrl}" alt="Person #${p.personId} Crop">
                    <div class="crop-zoom-hint">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                        <span>Enlarge</span>
                    </div>
                </div>` : ''}

                <div class="v-person-attrs">
                    <div class="v-attr-group">
                        <div class="v-attr-group-title">Appearance</div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Sex / Presentation:</span>
                            <span class="v-attr-val">${p.appearanceSex} ${p.appearanceConfidence > 0 ? `(${(p.appearanceConfidence * 100).toFixed(0)}%)` : ''}</span>
                        </div>
                    </div>

                    <div class="v-attr-group">
                        <div class="v-attr-group-title">Clothing Attributes</div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Upper Wear:</span>
                            <span class="v-attr-val">${formatClothing(p.upperColor, p.upperType)}</span>
                        </div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Lower Wear:</span>
                            <span class="v-attr-val">${formatClothing(p.lowerColor, p.lowerType)}</span>
                        </div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Footwear:</span>
                            <span class="v-attr-val">${formatStatusText(p.shoesType)}</span>
                        </div>
                    </div>

                    <div class="v-attr-group">
                        <div class="v-attr-group-title">Accessories</div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Wristwatch:</span>
                            <span class="v-attr-val">${formatStatusText(p.watchDetails || p.watchStatus)}</span>
                        </div>
                    </div>
                </div>
            </div>

            ${renderRegionalBrandPanel(p)}
        `;
        list.appendChild(card);
    });
}

function clearOverlayCanvas() {
    const canvas = document.getElementById('overlayCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    const list = document.getElementById('detectionsList');
    if (list) list.innerHTML = '';
}

function clearSelectedImage() {
    currentSelectedFile = null;
    const previewArea = document.getElementById('previewArea');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const pipelineResultsArea = document.getElementById('pipelineResultsArea');

    if (fileInput) fileInput.value = '';
    if (imagePreview) imagePreview.src = '';
    clearOverlayCanvas();
    if (pipelineResultsArea) pipelineResultsArea.style.display = 'none';
    if (previewArea) previewArea.style.display = 'none';
    if (dropzone) dropzone.style.display = 'block';
    hideNotification();
}

// -------------------------------------------------------------
// 2. TAB 2 — ANALYZE VIDEO (Multi-Person Sequential Tracking)
// -------------------------------------------------------------

function setupVideoDropzone() {
    const dropzone = document.getElementById('videoDropzone');
    if (!dropzone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = 'var(--primary-color)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = 'var(--border-color)';
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedVideo(files[0]);
        }
    }, false);
}

function handleVideoFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleSelectedVideo(files[0]);
    }
}

function handleSelectedVideo(file) {
    hideNotification();

    const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
    const fileName = file.name.toLowerCase();
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));

    if (!hasValidExtension) {
        showNotification('Invalid video format. Supported formats: .mp4, .mov, .avi, .mkv, .webm', 'error');
        return;
    }

    const maxSizeBytes = 100 * 1024 * 1024; // 100 MB
    if (file.size > maxSizeBytes) {
        showNotification(`Video is too large (${formatFileSize(file.size)}). Maximum allowed size is 100 MB.`, 'error');
        return;
    }

    currentSelectedVideoFile = file;

    const dropzone = document.getElementById('videoDropzone');
    const configBar = document.getElementById('videoConfigBar');
    const nameEl = document.getElementById('videoSelectedName');
    const sizeEl = document.getElementById('videoSelectedSize');
    const resultsArea = document.getElementById('videoResultsArea');

    if (nameEl) nameEl.textContent = file.name;
    if (sizeEl) sizeEl.textContent = `(${formatFileSize(file.size)})`;
    if (resultsArea) resultsArea.style.display = 'none';

    if (dropzone) dropzone.style.display = 'none';
    if (configBar) configBar.style.display = 'flex';
}

function clearSelectedVideo() {
    currentSelectedVideoFile = null;
    const dropzone = document.getElementById('videoDropzone');
    const configBar = document.getElementById('videoConfigBar');
    const progressArea = document.getElementById('videoProgressArea');
    const resultsArea = document.getElementById('videoResultsArea');
    const fileInput = document.getElementById('videoFileInput');

    if (fileInput) fileInput.value = '';
    if (progressArea) progressArea.style.display = 'none';
    if (resultsArea) resultsArea.style.display = 'none';
    if (configBar) configBar.style.display = 'none';
    if (dropzone) dropzone.style.display = 'block';
    hideNotification();
}

async function onAnalyzeVideoClicked() {
    if (!currentSelectedVideoFile) {
        showNotification('Please select a video file first.', 'error');
        return;
    }

    const btn = document.getElementById('btnAnalyzeVideo');
    const progressArea = document.getElementById('videoProgressArea');
    const resultsArea = document.getElementById('videoResultsArea');
    const fpsSelect = document.getElementById('videoFpsSelect');
    const targetFps = fpsSelect ? parseFloat(fpsSelect.value) : 3.0;

    try {
        btn.disabled = true;
        progressArea.style.display = 'block';
        resultsArea.style.display = 'none';
        hideNotification();

        const formData = new FormData();
        formData.append('file', currentSelectedVideoFile);
        formData.append('analysisFps', targetFps.toString());

        const response = await fetch('/api/analysis/video', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            const errorMsg = data.errorMessage || `Video analysis error (${response.status}: ${response.statusText})`;
            showNotification(errorMsg, 'error');
            return;
        }

        renderVideoResults(data);
        resultsArea.style.display = 'block';
        showNotification(`Video analysis complete! Identified ${data.summary?.totalUniquePersons || 0} unique person(s).`, 'success');
    } catch (err) {
        console.error('Video processing error:', err);
        showNotification(`Error during video analysis: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        progressArea.style.display = 'none';
    }
}

function renderVideoResults(data) {
    const summary = data.summary || {};
    const persons = data.persons || [];

    // Summary banner
    document.getElementById('vMetricPersons').textContent = summary.totalUniquePersons || persons.length;
    document.getElementById('vMetricDuration').textContent = `${(summary.videoDurationSeconds || 0).toFixed(1)}s`;
    document.getElementById('vMetricFrames').textContent = summary.sampledFramesAnalyzed || 0;
    document.getElementById('vMetricFps').textContent = `${(summary.analysisFps || 3.0).toFixed(1)} FPS`;
    document.getElementById('vMetricElapsed').textContent = `${(summary.totalElapsedMs || 0).toFixed(0)} ms`;

    // Person cards
    const grid = document.getElementById('videoPersonsList');
    grid.innerHTML = '';

    if (persons.length === 0) {
        grid.innerHTML = '<div class="no-detections-message">No persons detected in video.</div>';
        return;
    }

    persons.forEach(p => {
        const color = getPersonColor(p.personId);
        const card = document.createElement('div');
        card.className = 'video-person-card';
        card.style.borderTop = `4px solid ${color}`;
        const personIdFormatted = String(p.personId).padStart(2, '0');

        const statusBadge = p.trackStatus === 'ActiveUntilEnd'
            ? '<span class="state-badge state-stable">Active At End</span>'
            : '<span class="state-badge state-insufficient">Finalized</span>';

        const switchWarning = (p.trackingDiagnostics && p.trackingDiagnostics.possibleIdSwitch)
            ? '<span class="state-badge state-notvisible">ID Switch Warning</span>'
            : '';

        card.innerHTML = `
            <div class="v-person-header">
                <div class="person-tag-wrap">
                    <span class="person-color-dot" style="background-color: ${color}; color: ${color};"></span>
                    <span class="v-person-id" style="color: ${color};">PERSON ${personIdFormatted}</span>
                </div>
                <div style="display: flex; gap: 0.4rem; align-items: center;">${switchWarning} ${statusBadge}</div>
            </div>
            
            <div class="v-person-timestamps">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                <span>Seen: <strong>${p.firstSeenTimestamp}</strong> &rarr; <strong>${p.lastSeenTimestamp}</strong> (${p.framesObserved} observations)</span>
            </div>

            <div class="v-person-body">
                ${p.bestCropPreviewUrl ? `
                <div class="v-person-crop" title="Click to enlarge best observation crop" onclick="openImageModal('${p.bestCropPreviewUrl}', 'Person #${p.personId} Best Crop', '${p.appearanceSex} (${p.appearanceState}) • Seen: ${p.firstSeenTimestamp} → ${p.lastSeenTimestamp}')">
                    <img src="${p.bestCropPreviewUrl}" alt="Person #${p.personId} Best Crop">
                    <div class="crop-zoom-hint">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                        <span>Enlarge</span>
                    </div>
                </div>` : ''}

                <div class="v-person-attrs">
                    <div class="v-attr-group">
                        <div class="v-attr-group-title">Appearance</div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Sex / Presentation:</span>
                            <span class="v-attr-val">${p.appearanceSex} <span class="state-badge state-${(p.appearanceState || 'analyzing').toLowerCase()}">${p.appearanceState}</span></span>
                        </div>
                    </div>

                    <div class="v-attr-group">
                        <div class="v-attr-group-title">Clothing Attributes</div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Upper Wear:</span>
                            <span class="v-attr-val">${formatClothing(p.upperColor, p.upperType)}</span>
                        </div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Lower Wear:</span>
                            <span class="v-attr-val">${formatClothing(p.lowerColor, p.lowerType)}</span>
                        </div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Footwear:</span>
                            <span class="v-attr-val">${formatStatusText(p.shoesType)}</span>
                        </div>
                    </div>

                    <div class="v-attr-group">
                        <div class="v-attr-group-title">Accessories</div>
                        <div class="v-attr-row">
                            <span class="v-attr-label">Wristwatch:</span>
                            <span class="v-attr-val">${formatStatusText(p.watchDetails || p.watchStatus)}</span>
                        </div>
                    </div>
                </div>
            </div>

            ${renderRegionalBrandPanel(p)}
            ${renderBestFramesAccordion(p)}
        `;

        grid.appendChild(card);
    });
}

function renderBestFramesAccordion(p) {
    const appFrames = p.bestAppearanceFrames || [];
    const upperFrames = p.bestUpperFrames || [];
    const lowerFrames = p.bestLowerFrames || [];
    const shoesFrames = p.bestShoesFrames || [];

    if (appFrames.length === 0 && upperFrames.length === 0) return '';

    return `
        <div class="best-frames-wrap">
            <div class="best-frames-title">Top-5 Best Quality Observations (Pool):</div>
            <div>
                ${upperFrames.slice(0, 3).map(f => `<span class="best-frame-pill">#${f.frameIndex}: ${f.prediction} (Q:${(f.qualityScore * 100).toFixed(0)}%)</span>`).join('')}
            </div>
        </div>
    `;
}

// -------------------------------------------------------------
// 3. TAB 3 — LIVE CAMERA STREAMING & MULTI-PERSON TRACKING
// -------------------------------------------------------------

let cameraStream = null;
let isCameraRunning = false;
let cameraIntervalId = null;
let currentCameraFacingMode = 'user'; // 'user' or 'environment'
let currentFrameIndex = 0;
let isProcessingFrame = false;
let lastFrameTime = performance.now();
let fpsCounter = 0;
let lastFpsUpdate = performance.now();

async function populateCameraDevices() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        const select = document.getElementById('cameraSelect');
        if (!select) return;

        select.innerHTML = '<option value="">Default Web Camera</option>';
        videoDevices.forEach((device, index) => {
            const opt = document.createElement('option');
            opt.value = device.deviceId;
            opt.textContent = device.label || `Camera ${index + 1}`;
            select.appendChild(opt);
        });

        if (videoDevices.length > 1) {
            const flipBtn = document.getElementById('btnFlipCamera');
            if (flipBtn) flipBtn.style.display = 'inline-block';
        }
    } catch (e) {
        console.warn('Unable to enumerate camera devices:', e);
    }
}

async function onCameraDeviceChanged() {
    if (!isCameraRunning) return;

    // Stop the active stream before opening the newly selected device.
    stopCamera();
    await startCamera();
}

async function startCamera() {
    try {
        const video = document.getElementById('cameraVideo');
        const select = document.getElementById('cameraSelect');
        const placeholder = document.getElementById('cameraPlaceholder');
        const btnStart = document.getElementById('btnStartCamera');
        const btnStop = document.getElementById('btnStopCamera');
        const btnFlip = document.getElementById('btnFlipCamera');

        const deviceId = select ? select.value : '';
        const constraints = {
            video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: currentCameraFacingMode, width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = cameraStream;
        await video.play();

        isCameraRunning = true;
        currentFrameIndex = 0;

        if (placeholder) placeholder.style.display = 'none';
        if (btnStart) btnStart.style.display = 'none';
        if (btnStop) btnStop.style.display = 'inline-block';
        if (btnFlip) btnFlip.style.display = 'inline-block';

        // Start frame capture loop (every ~100ms / 10 FPS)
        cameraIntervalId = setInterval(captureAndAnalyzeLiveFrame, 100);
        showNotification('Live Camera started.', 'success');
    } catch (err) {
        console.error('Camera startup failed:', err);
        showNotification(`Camera access failed: ${err.message}`, 'error');
    }
}

function stopCamera() {
    isCameraRunning = false;
    if (cameraIntervalId) {
        clearInterval(cameraIntervalId);
        cameraIntervalId = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    const video = document.getElementById('cameraVideo');
    const placeholder = document.getElementById('cameraPlaceholder');
    const btnStart = document.getElementById('btnStartCamera');
    const btnStop = document.getElementById('btnStopCamera');
    const btnFlip = document.getElementById('btnFlipCamera');
    const canvas = document.getElementById('cameraOverlayCanvas');

    if (video) video.srcObject = null;
    if (placeholder) placeholder.style.display = 'flex';
    if (btnStart) btnStart.style.display = 'inline-block';
    if (btnStop) btnStop.style.display = 'none';
    if (btnFlip) btnFlip.style.display = 'none';

    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Reset backend live tracking
    fetch('/api/live/reset', { method: 'POST' }).catch(console.warn);

    document.getElementById('trackedPersonsList').innerHTML = '<div class="empty-tracks-notice">Camera stopped.</div>';
    document.getElementById('activeTrackBadge').textContent = '0 Active';
}

function flipCamera() {
    currentCameraFacingMode = currentCameraFacingMode === 'user' ? 'environment' : 'user';
    stopCamera();
    startCamera();
}

async function captureAndAnalyzeLiveFrame() {
    if (!isCameraRunning || isProcessingFrame) return;

    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraOverlayCanvas');

    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    isProcessingFrame = true;
    currentFrameIndex++;

    try {
        // Only resize canvas if video dimensions changed (resizing resets canvas bitmap and causes flickering)
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        // In-memory offscreen canvas capture to JPEG Blob
        const offscreen = document.createElement('canvas');
        offscreen.width = video.videoWidth;
        offscreen.height = video.videoHeight;
        const offCtx = offscreen.getContext('2d');
        offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

        offscreen.toBlob(async (blob) => {
            if (!blob) {
                isProcessingFrame = false;
                return;
            }

            const sendTime = performance.now();
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            formData.append('frameIndex', currentFrameIndex.toString());

            try {
                const response = await fetch('/api/live/process-frame', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                const latency = performance.now() - sendTime;

                if (response.ok && data.success) {
                    renderLiveOverlay(data.persons, video.videoWidth, video.videoHeight);
                    renderLivePersonsCards(data.persons);
                    renderLiveDiagnostics(data.persons);

                    // Update stats
                    fpsCounter++;
                    const now = performance.now();
                    if (now - lastFpsUpdate >= 1000) {
                        document.getElementById('cameraFps').textContent = `${fpsCounter} FPS`;
                        fpsCounter = 0;
                        lastFpsUpdate = now;
                    }
                    document.getElementById('cameraLatency').textContent = `${latency.toFixed(0)}ms`;
                    document.getElementById('cameraTrackCount').textContent = data.persons.length.toString();
                    document.getElementById('activeTrackBadge').textContent = `${data.persons.length} Active`;
                }
            } catch (postErr) {
                console.warn('Frame processing dropped:', postErr);
            } finally {
                isProcessingFrame = false;
            }
        }, 'image/jpeg', 0.75);

    } catch (err) {
        console.error('Frame capture exception:', err);
        isProcessingFrame = false;
    }
}

function renderLiveOverlay(persons, canvasWidth, canvasHeight) {
    const canvas = document.getElementById('cameraOverlayCanvas');
    if (!canvas) return;

    if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!persons || persons.length === 0) return;

    persons.forEach((p, idx) => {
        const dispId = p.displayId || (idx + 1);
        const box = p.boundingBox;
        const color = getPersonColor(dispId);

        // Bounding Box
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(3, Math.round(canvasWidth / 250));
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        // Label (Starts from Person #1 for active person on screen)
        const orientation = p.visibility?.orientation || '';
        const label = `Person #${dispId} (${(p.detectionConfidence * 100).toFixed(0)}%) [${orientation}]`;
        const fontSize = Math.max(13, Math.round(canvasWidth / 45));
        ctx.font = `600 ${fontSize}px Inter, sans-serif`;

        const metrics = ctx.measureText(label);
        const padding = 4;
        const labelH = fontSize + padding * 2;
        const labelW = metrics.width + padding * 2;
        let labelY = box.y - labelH;
        if (labelY < 0) labelY = box.y;

        ctx.fillStyle = color;
        ctx.fillRect(box.x, labelY, labelW, labelH);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, box.x + padding, labelY + fontSize);

        // Gesture badge on box if hand raised
        const isLeftRaised = p.visibility?.leftHandRaised;
        const isRightRaised = p.visibility?.rightHandRaised;
        if (isLeftRaised || isRightRaised) {
            const gestureText = (isLeftRaised && isRightRaised) ? '👐 BOTH HANDS RAISED' : (isLeftRaised ? '✋ LEFT HAND RAISED' : '✋ RIGHT HAND RAISED');
            const gFontSize = Math.max(11, Math.round(canvasWidth / 55));
            ctx.font = `bold ${gFontSize}px Inter, sans-serif`;
            const gMetrics = ctx.measureText(gestureText);
            const gH = gFontSize + 8;
            const gW = gMetrics.width + 12;
            const gY = box.y + box.height + 4;
            ctx.fillStyle = 'rgba(245, 185, 66, 0.95)';
            ctx.fillRect(box.x, gY, gW, gH);
            ctx.fillStyle = '#070B14';
            ctx.fillText(gestureText, box.x + 6, gY + gFontSize + 1);
        }

        // Render Pose Keypoints with Wrist Labels
        if (p.keypoints && p.keypoints.length > 0) {
            p.keypoints.forEach(k => {
                if (k.confidence >= 0.35) {
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(k.x, k.y, 4, 0, 2 * Math.PI);
                    ctx.fill();

                    // Distinct wrist indicators (Kp 9 = LeftWrist, Kp 10 = RightWrist)
                    if (k.index === 9 || k.index === 10) {
                        const isL = k.index === 9;
                        const labelText = isL ? 'LW' : 'RW';
                        ctx.fillStyle = isL ? '#4F8CFF' : '#FF5C6C';
                        ctx.beginPath();
                        ctx.arc(k.x, k.y, 6, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.fillStyle = '#ffffff';
                        ctx.font = 'bold 9px sans-serif';
                        ctx.fillText(labelText, k.x + 8, k.y + 3);
                    }
                }
            });
        }
    });
}

let selectedLivePersonId = null;
let currentLivePersonsList = [];

function selectLivePerson(personId) {
    selectedLivePersonId = personId;
    if (currentLivePersonsList && currentLivePersonsList.length > 0) {
        renderLivePersonsCards(currentLivePersonsList);
    }
}

function renderLivePersonsCards(persons) {
    const container = document.getElementById('trackedPersonsList');
    const tabsContainer = document.getElementById('personTabsContainer');
    const tabsBar = document.getElementById('personTabsBar');
    if (!container) return;

    currentLivePersonsList = persons || [];

    if (!persons || persons.length === 0) {
        if (tabsContainer) tabsContainer.style.display = 'none';
        container.innerHTML = '<div class="empty-tracks-notice">Waiting for person detections...</div>';
        selectedLivePersonId = null;
        return;
    }

    // 1. Maintain or default selected person ID
    const personIds = persons.map(p => p.personId);
    if (selectedLivePersonId === null || !personIds.includes(selectedLivePersonId)) {
        selectedLivePersonId = persons[0].personId;
    }

    // 2. Render Top Person Tabs (Display ID starts from 01 for active persons)
    if (tabsContainer && tabsBar) {
        tabsContainer.style.display = 'block';
        tabsBar.innerHTML = persons.map((p, idx) => {
            const dispId = p.displayId || (idx + 1);
            const color = getPersonColor(dispId);
            const isSelected = p.personId === selectedLivePersonId;
            const pIdFmt = String(dispId).padStart(2, '0');
            const summaryTag = p.appearanceSex !== 'Unknown' && p.appearanceSex !== 'Insufficient Evidence'
                ? p.appearanceSex
                : (p.visibility?.orientation || 'Active');

            return `
                <button class="person-tab-btn ${isSelected ? 'active' : ''}" 
                        style="--tab-color: ${color};" 
                        onclick="selectLivePerson(${p.personId})"
                        title="Click to inspect Person #${dispId} (Track #${p.personId})">
                    <span class="person-color-dot" style="background-color: ${color}; width: 8px; height: 8px; border-radius: 50%;"></span>
                    <span class="person-tab-name" style="color: ${isSelected ? color : 'inherit'};">PERSON ${pIdFmt}</span>
                    <span class="person-tab-pill">${summaryTag}</span>
                </button>
            `;
        }).join('');
    }

    // 3. Render Inspector Card for the Selected Person
    const selectedPerson = persons.find(p => p.personId === selectedLivePersonId) || persons[0];
    const selectedDispId = selectedPerson.displayId || (persons.indexOf(selectedPerson) + 1);
    const color = getPersonColor(selectedDispId);
    const pIdFormatted = String(selectedDispId).padStart(2, '0');

    const stabilityTag = formatStateBadge(selectedPerson.appearanceState, selectedPerson.isFullyStable);

    const switchWarning = (selectedPerson.trackingDiagnostics && selectedPerson.trackingDiagnostics.possibleIdSwitch)
        ? '<span class="state-badge state-notvisible">ID Switch Warning</span>'
        : '';

    container.innerHTML = `
        <div class="tracked-person-card" style="border-left: 5px solid ${color};">
            <div class="person-card-header">
                <div class="person-tag-wrap">
                    <span class="person-color-dot" style="background-color: ${color};"></span>
                    <span class="person-id-label" style="color: ${color};">PERSON ${pIdFormatted}</span>
                    <span class="detection-conf-badge">${(selectedPerson.detectionConfidence * 100).toFixed(0)}% Conf</span>
                </div>
                <div class="person-status-wrap">${switchWarning} ${stabilityTag}</div>
            </div>

            <div class="person-card-body">
                <div class="person-attr-line">
                    <span class="attr-title">Appearance:</span>
                    <strong class="attr-val">${selectedPerson.appearanceSex}</strong>
                </div>
                <div class="person-attr-line">
                    <span class="attr-title">Upper Wear:</span>
                    <strong class="attr-val">${formatClothing(selectedPerson.upperColor, selectedPerson.upperType)}</strong>
                </div>
                <div class="person-attr-line">
                    <span class="attr-title">Lower Wear:</span>
                    <strong class="attr-val">${formatClothing(selectedPerson.lowerColor, selectedPerson.lowerType)}</strong>
                </div>
                <div class="person-attr-line">
                    <span class="attr-title">Footwear:</span>
                    <strong class="attr-val">${formatStatusText(selectedPerson.shoesType)}</strong>
                </div>
                <div class="person-attr-line">
                    <span class="attr-title">Wristwatch:</span>
                    <strong class="attr-val">${formatStatusText(selectedPerson.watchDetails || selectedPerson.watchStatus)}</strong>
                </div>
                <div class="person-attr-line">
                    <span class="attr-title">Gestures / Hands:</span>
                    <strong class="attr-val">
                        ${selectedPerson.visibility?.leftHandRaised ? '<span style="background:var(--accent-amber); color:#070B14; font-weight:700; padding:2px 6px; border-radius:4px; font-size:0.75rem; margin-right:4px;">Left Hand Raised</span>' : ''}
                        ${selectedPerson.visibility?.rightHandRaised ? '<span style="background:var(--accent-amber); color:#070B14; font-weight:700; padding:2px 6px; border-radius:4px; font-size:0.75rem;">Right Hand Raised</span>' : ''}
                        ${(!selectedPerson.visibility?.leftHandRaised && !selectedPerson.visibility?.rightHandRaised) ? '<span style="color:var(--text-muted); font-size:0.8rem;">Hands Down</span>' : ''}
                    </strong>
                </div>
                <div class="person-attr-line">
                    <span class="attr-title">Orientation:</span>
                    <span class="orientation-chip">${selectedPerson.visibility?.orientation || 'Uncertain'}</span>
                </div>
                ${renderRegionalBrandPanel(selectedPerson)}
            </div>
        </div>
    `;
}

function renderLiveDiagnostics(persons) {
    const tbody = document.getElementById('debugTableBody');
    if (!tbody) return;

    if (!persons || persons.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No active tracks.</td></tr>';
        return;
    }

    tbody.innerHTML = '';

    persons.forEach((p, idx) => {
        const dispId = p.displayId || (idx + 1);
        const q = p.qualityScores || {};
        const v = p.visibility || {};
        const tr = document.createElement('tr');

        const clipping = [
            v.headClipped ? 'Head' : '',
            v.upperClipped ? 'Upper' : '',
            v.lowerClipped ? 'Lower' : '',
            v.feetClipped ? 'Feet' : ''
        ].filter(Boolean).join(', ') || 'None';

        tr.innerHTML = `
            <td><strong>#${dispId}</strong> <span style="font-size:0.72rem; color:var(--text-muted);">(Track #${p.personId})</span></td>
            <td>Age: ${p.trackingDiagnostics?.trackAgeFrames || 1}f<br/>IoU: ${(p.trackingDiagnostics?.lastMatchedIoU || 1).toFixed(2)}</td>
            <td><span class="orientation-chip">${v.orientation || 'Uncertain'}</span></td>
            <td>
                App: ${(q.appearance || 0).toFixed(2)}<br/>
                Up: ${(q.upper || 0).toFixed(2)}<br/>
                Low: ${(q.lower || 0).toFixed(2)}
            </td>
            <td>Conf: ${(p.detectionConfidence * 100).toFixed(0)}%</td>
            <td>${clipping}</td>
            <td>
                App: ${p.appearanceState}<br/>
                Up: ${p.upperTypeState}
            </td>
            <td>
                ${(p.bestFrames?.upper || []).slice(0, 2).map(f => `#${f.frameIndex}: ${f.prediction}`).join('<br/>') || '-'}
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function toggleDebugAccordion() {
    const content = document.getElementById('debugContent');
    const arrow = document.getElementById('debugArrow');
    if (!content) return;

    if (content.style.display === 'none') {
        content.style.display = 'block';
        if (arrow) arrow.textContent = '▲';
    } else {
        content.style.display = 'none';
        if (arrow) arrow.textContent = '▼';
    }
}

// -------------------------------------------------------------
// 4. BENCHMARK MODE MODAL & EVALUATION
// -------------------------------------------------------------

function openBenchmarkModal() {
    const modal = document.getElementById('benchmarkModal');
    if (modal) modal.style.display = 'flex';
}

function closeBenchmarkModal() {
    const modal = document.getElementById('benchmarkModal');
    if (modal) modal.style.display = 'none';
}

async function submitGroundTruth() {
    const personId = parseInt(document.getElementById('gtPersonId').value, 10);
    const sex = document.getElementById('gtSex').value;
    const upType = document.getElementById('gtUpperType').value;
    const upColor = document.getElementById('gtUpperColor').value;
    const lowType = document.getElementById('gtLowerType').value;
    const shoes = document.getElementById('gtShoes').value;

    const payload = {
        personId: personId,
        expectedSex: sex,
        expectedUpperType: upType,
        expectedUpperColor: upColor,
        expectedLowerType: lowType,
        expectedShoes: shoes
    };

    try {
        const resp = await fetch('/api/benchmark/ground-truth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (resp.ok && data.success) {
            showNotification(`Ground truth saved for Person #${personId}.`, 'success');
        }
    } catch (e) {
        showNotification(`Error saving ground truth: ${e.message}`, 'error');
    }
}

async function resetGroundTruths() {
    try {
        await fetch('/api/benchmark/reset', { method: 'POST' });
        showNotification('Benchmark ground truths reset.', 'success');
        document.getElementById('benchmarkReportCard').style.display = 'none';
    } catch (e) {
        showNotification(`Reset failed: ${e.message}`, 'error');
    }
}

async function evaluateBenchmark() {
    try {
        const resp = await fetch('/api/benchmark/evaluation');
        const data = await resp.json();

        const reportCard = document.getElementById('benchmarkReportCard');
        const overallScore = document.getElementById('benchmarkOverallScore');
        const tbody = document.getElementById('benchmarkTableBody');
        const timestamp = document.getElementById('benchmarkEvalTimestamp');

        if (timestamp) timestamp.textContent = `Evaluated at ${new Date().toLocaleTimeString()}`;
        if (overallScore) overallScore.textContent = `${(data.overallAccuracyPercent || 0).toFixed(1)}%`;

        tbody.innerHTML = '';
        (data.attributeMetrics || []).forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${m.attributeName}</strong></td>
                <td>${m.totalEvaluated}</td>
                <td style="color: var(--success-color);">${m.correctCount}</td>
                <td style="color: var(--error-color);">${m.incorrectCount}</td>
                <td><strong>${(m.accuracyPercent || 0).toFixed(1)}%</strong></td>
                <td>${m.insufficientEvidenceCount}</td>
            `;
            tbody.appendChild(tr);
        });

        reportCard.style.display = 'block';
    } catch (e) {
        showNotification(`Benchmark evaluation failed: ${e.message}`, 'error');
    }
}

// -------------------------------------------------------------
// 5. DEVELOPER DIAGNOSTICS: LEGACY PULC PAR TEST HANDLERS
// -------------------------------------------------------------

function setupParDropzone() {
    const dropzone = document.getElementById('parDropzone');
    if (!dropzone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = 'var(--primary-color)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = 'var(--border-color)';
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleParSelectedFile(files[0]);
        }
    }, false);
}

function handleParFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleParSelectedFile(files[0]);
    }
}

function handleParSelectedFile(file) {
    currentParSelectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewArea = document.getElementById('parPreviewArea');
        const dropzone = document.getElementById('parDropzone');
        const imagePreview = document.getElementById('parImagePreview');
        const filenameEl = document.getElementById('parFilename');
        const resultsCard = document.getElementById('parResultsCard');

        if (imagePreview) {
            imagePreview.src = e.target.result;
            imagePreview.style.cursor = 'pointer';
            imagePreview.title = 'Click to preview full image';
            imagePreview.onclick = () => openImageModal(imagePreview.src, 'PAR Person Crop Preview', file.name);
        }
        if (filenameEl) filenameEl.textContent = `${file.name} (${formatFileSize(file.size)})`;
        if (resultsCard) resultsCard.style.display = 'none';

        if (dropzone) dropzone.style.display = 'none';
        if (previewArea) previewArea.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearParSelectedImage() {
    currentParSelectedFile = null;
    const previewArea = document.getElementById('parPreviewArea');
    const dropzone = document.getElementById('parDropzone');
    const fileInput = document.getElementById('parFileInput');
    const resultsCard = document.getElementById('parResultsCard');

    if (fileInput) fileInput.value = '';
    if (resultsCard) resultsCard.style.display = 'none';
    if (previewArea) previewArea.style.display = 'none';
    if (dropzone) dropzone.style.display = 'block';
}

async function onAnalyzeParClicked() {
    if (!currentParSelectedFile) return;

    const btn = document.getElementById('btnAnalyzePar');
    const btnText = document.getElementById('btnAnalyzeParText');
    const resultsCard = document.getElementById('parResultsCard');

    try {
        btn.disabled = true;
        btnText.textContent = 'Running PULC Inference...';

        const formData = new FormData();
        formData.append('file', currentParSelectedFile);

        const response = await fetch('/api/analysis/test-par', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            document.getElementById('parElapsedBadge').textContent = `${data.elapsedMs.toFixed(1)} ms`;
            const attrs = data.attributes?.attributes || {};

            document.getElementById('valAppearanceSex').textContent = attrs.Appearance?.value || 'Unknown';
            document.getElementById('valAgeGroup').textContent = attrs.Age?.value || 'Adult';
            document.getElementById('valOrientation').textContent = attrs.Orientation?.value || 'Front';
            document.getElementById('valSleeveType').textContent = attrs.Sleeve?.value || 'Short';
            document.getElementById('valUpperWear').textContent = attrs.UpperWear?.value || 'T-Shirt';
            document.getElementById('valLowerWear').textContent = attrs.LowerWear?.value || 'Trousers';
            document.getElementById('valFootwear').textContent = attrs.Footwear?.value || 'Shoes';

            resultsCard.style.display = 'block';
        }
    } catch (e) {
        showNotification(`PULC test error: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Run PULC Inference';
    }
}

// -------------------------------------------------------------
// 6. HEALTH & UTILITIES
// -------------------------------------------------------------

async function checkHealth() {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');

    try {
        const resp = await fetch('/api/analysis/health');
        const data = await resp.json();

        if (resp.ok && data.status === 'Online') {
            dot.className = 'status-dot green';
            text.textContent = 'AI Pipeline Ready';
        } else {
            dot.className = 'status-dot orange';
            text.textContent = 'Degraded';
        }
    } catch (e) {
        dot.className = 'status-dot red';
        text.textContent = 'Offline';
    }
}

function showNotification(msg, type = 'info') {
    const banner = document.getElementById('notificationBanner');
    const msgEl = document.getElementById('notificationMessage');
    const iconEl = document.getElementById('notificationIcon');

    if (!banner || !msgEl) return;

    msgEl.textContent = msg;
    banner.className = `notification-banner ${type}`;
    if (iconEl) {
        if (type === 'error') {
            iconEl.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
        } else if (type === 'success') {
            iconEl.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
        } else {
            iconEl.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
        }
    }

    banner.style.display = 'flex';
    setTimeout(() => {
        banner.style.display = 'none';
    }, 5000);
}

function hideNotification() {
    const banner = document.getElementById('notificationBanner');
    if (banner) banner.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// -------------------------------------------------------------
// 7. LIGHTBOX IMAGE MODAL PREVIEW
// -------------------------------------------------------------

function openImageModal(src, title = 'Person Preview', caption = '') {
    if (!src) return;
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('imageModalImg');
    const modalTitle = document.getElementById('imageModalTitle');
    const modalCaption = document.getElementById('imageModalCaption');

    if (!modal || !modalImg) return;

    modalImg.src = src;
    if (modalTitle) modalTitle.textContent = title;
    if (modalCaption) modalCaption.innerHTML = caption;

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeImageModal(e) {
    if (e && e.target && e.target.classList.contains('image-modal-content')) {
        return;
    }
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeImageModal();
    }
});
