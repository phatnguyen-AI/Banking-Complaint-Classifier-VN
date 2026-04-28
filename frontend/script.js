/**
 * Banking Complaint Classifier - Frontend Application Logic
 *
 * Handles user interaction, API communication, and result display.
 * All API calls are async with error handling and graceful degradation
 * when the backend is unavailable (Render free tier cold starts).
 */

// -- Configuration --
const API_BASE_URL = 'https://llm-vhhs.onrender.com';
const API_ENDPOINT = `${API_BASE_URL}/predict`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/health`;

// -- DOM References --
// Cached at module level to avoid repeated lookups.
const textInput = document.getElementById('text-input');
const charCount = document.getElementById('char-count');
const classifyBtn = document.getElementById('classify-btn');
const exampleBtn = document.getElementById('example-btn');
const resultsSection = document.getElementById('results-section');
const predictedClass = document.getElementById('predicted-class');
const confidenceScore = document.getElementById('confidence-score');
const confidenceBar = document.getElementById('confidence-bar');
const newAnalysisBtn = document.getElementById('new-analysis-btn');
const apiStatus = document.getElementById('api-status');

// Sample complaints covering all 6 categories for quick testing.
const exampleTexts = [
    'The cua toi bi loi khong the thanh toan duoc',
    'Toi khong the dang nhap vao ung dung',
    'Giao dich chuyen tien that bai',
    'Toi muon vay tien mua nha',
    'Tai khoan cua toi co dau hieu bi xam nhap',
    'Lam sao de doi mat khau?'
];

// -- Initialization --
document.addEventListener('DOMContentLoaded', function() {
    checkApiStatus();
    setupEventListeners();
    updateCharCount();
});

// -- Event Binding --
// Null-checks on every element prevent runtime errors if the DOM
// structure changes or elements are conditionally rendered.
function setupEventListeners() {
    const form = document.getElementById('classification-form');
    if (form) form.addEventListener('submit', handleFormSubmit);
    if (textInput) textInput.addEventListener('input', updateCharCount);
    if (exampleBtn) exampleBtn.addEventListener('click', showExampleText);
    if (newAnalysisBtn) newAnalysisBtn.addEventListener('click', resetForm);
}

// -- API: Health Check --
// Called on page load and every 30s. Updates the status badge in the navbar.
async function checkApiStatus() {
    try {
        const response = await fetch(HEALTH_ENDPOINT);
        const data = await response.json();

        if (apiStatus) {
            const isOnline = response.ok && data.status === 'ok';
            apiStatus.textContent = isOnline ? 'Online' : 'Offline';
            apiStatus.className = isOnline ? 'badge bg-success' : 'badge bg-danger';
        }
    } catch (error) {
        // Network failure or CORS issue -- mark as offline.
        if (apiStatus) {
            apiStatus.textContent = 'Offline';
            apiStatus.className = 'badge bg-danger';
        }
    }
}

// -- API: Classification --
// Sends POST /predict with the complaint text. Returns {label, score}.
async function classifyText(text) {
    const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
    }

    return await response.json();
}

// -- Form Submission Handler --
async function handleFormSubmit(event) {
    event.preventDefault();
    if (!textInput) return;

    const text = textInput.value.trim();

    // Client-side validation: reject short inputs before hitting the API.
    if (text.length < 10) {
        showError('Please enter at least 10 characters.');
        return;
    }

    // Disable submit button to prevent duplicate requests.
    if (classifyBtn) {
        classifyBtn.disabled = true;
        classifyBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span> Processing...';
    }

    // Show loading modal. Using getOrCreateInstance avoids stale reference
    // issues that occur when caching the Modal instance at module level.
    const modalEl = document.getElementById('loading-modal');
    const loadingModal = bootstrap.Modal.getOrCreateInstance(modalEl);
    loadingModal.show();

    try {
        const result = await classifyText(text);

        // Brief delay ensures the Bootstrap modal transition completes
        // before we attempt to hide it, preventing a stuck backdrop.
        await new Promise(resolve => setTimeout(resolve, 500));
        loadingModal.hide();

        displayResults(result);

        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        loadingModal.hide();
        showError('Classification failed. The API may be starting up -- please retry in 10 seconds.');
    } finally {
        // Re-enable the submit button regardless of success or failure.
        if (classifyBtn) {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = '<i class="fas fa-search me-2"></i>Classify';
        }

        // Workaround: Bootstrap occasionally orphans the backdrop element
        // when hide() is called during a transition. Clean up manually.
        setTimeout(() => {
            const backdrops = document.querySelectorAll('.modal-backdrop');
            if (backdrops.length > 0 && !document.body.classList.contains('modal-open')) {
                backdrops.forEach(bd => bd.remove());
            }
        }, 500);
    }
}

// -- Character Counter --
// Visual feedback: red below minimum threshold, green when valid.
function updateCharCount() {
    if (!textInput || !charCount) return;
    const count = textInput.value.length;
    charCount.textContent = count;
    charCount.style.color = count < 10 ? 'var(--danger-color)' : 'var(--success-color)';
}

// -- Example Text Loader --
// Picks a random sample from exampleTexts to populate the textarea.
function showExampleText() {
    if (!textInput) return;
    textInput.value = exampleTexts[Math.floor(Math.random() * exampleTexts.length)];
    updateCharCount();
    textInput.focus();
}

// -- Result Display --
// Renders the API response and color-codes the confidence bar:
//   >= 80% green, >= 60% yellow, < 60% red.
function displayResults(result) {
    if (!predictedClass || !confidenceScore || !confidenceBar) return;

    predictedClass.textContent = result.label || 'Unknown';
    confidenceScore.textContent = `${(result.score * 100).toFixed(1)}%`;
    confidenceBar.style.width = `${(result.score * 100)}%`;

    if (resultsSection) {
        resultsSection.classList.remove('d-none');
        resultsSection.classList.add('fade-in');
    }

    // Color thresholds for confidence visualization.
    confidenceBar.className = 'progress-bar';
    if (result.score >= 0.8) {
        confidenceBar.classList.add('bg-success');
    } else if (result.score >= 0.6) {
        confidenceBar.classList.add('bg-warning');
    } else {
        confidenceBar.classList.add('bg-danger');
    }
}

// -- Form Reset --
function resetForm() {
    if (!textInput) return;
    textInput.value = '';
    updateCharCount();
    if (resultsSection) resultsSection.classList.add('d-none');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    textInput.focus();
}

// -- Toast Notification --
// Non-blocking error display. Falls back to setTimeout if Bootstrap
// JS fails to load (e.g., CDN outage).
function showError(message) {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();

    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = `
        <div class="toast align-items-center text-white bg-danger border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    const toastEl = tempDiv.firstElementChild;
    toastContainer.appendChild(toastEl);

    if (typeof bootstrap !== 'undefined') {
        const bsToast = new bootstrap.Toast(toastEl);
        bsToast.show();
        toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
    } else {
        setTimeout(() => toastEl.remove(), 5000);
    }
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    container.style.zIndex = '1055';
    document.body.appendChild(container);
    return container;
}

// -- Periodic Health Polling --
// 30s interval matches the Render health check frequency.
// Keeps the status badge accurate without overloading the free-tier backend.
setInterval(checkApiStatus, 30000);