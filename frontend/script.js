document.addEventListener('DOMContentLoaded', () => {
    const symptomInput = document.getElementById('symptomInput');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const resultSection = document.getElementById('resultSection');
    const diseaseResult = document.getElementById('diseaseResult');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const loader = document.getElementById('loader');

    const API_URL = 'http://localhost:8000/predict';
    const TIMEOUT_MS = 30000; // 30 second timeout

    function showLoader() {
        loader.classList.remove('hidden');
        predictBtn.disabled = true;
    }

    function hideLoader() {
        loader.classList.add('hidden');
        predictBtn.disabled = false;
    }

    function showError(message) {
        hideLoader();
        diseaseResult.textContent = 'Error';
        const confidenceContainer = document.querySelector('.confidence-container');
        confidenceContainer.style.display = 'none';
        resultSection.classList.remove('hidden');

        // Show error in disclaimer
        const disclaimer = document.querySelector('.disclaimer p');
        disclaimer.textContent = message;
        document.querySelector('.disclaimer').style.background = '#fef2f2';
        document.querySelector('.disclaimer').style.color = '#991b1b';

        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Fetch with timeout utility
    async function fetchWithTimeout(url, options, timeoutMs) {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, { ...options, signal: controller.signal });
            clearTimeout(id);
            return response;
        } catch (err) {
            clearTimeout(id);
            throw err;
        }
    }

    predictBtn.addEventListener('click', async () => {
        const symptoms = symptomInput.value.trim();

        if (!symptoms) {
            alert('Please enter some symptoms first.');
            return;
        }

        showLoader();

        try {
            const response = await fetchWithTimeout(
                API_URL,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symptoms }),
                },
                TIMEOUT_MS
            );

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();
            hideLoader();
            displayResult(data);

        } catch (error) {
            console.error('Prediction error:', error);
            if (error.name === 'AbortError') {
                showError('Request timed out after 30 seconds. The model may still be loading — please try again.');
            } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                showError('Cannot connect to backend at localhost:8000. Make sure the FastAPI server is running (run start_all.py).');
            } else {
                showError(`Error: ${error.message}`);
            }
        }
    });

    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        symptomInput.value = '';

        // Reset disclaimer styling
        const disclaimer = document.querySelector('.disclaimer');
        disclaimer.style.background = '';
        disclaimer.style.color = '';
        const disclaimerP = disclaimer.querySelector('p');
        disclaimerP.textContent = 'This is an AI prediction based on provided symptoms. Please consult a medical professional for an accurate diagnosis.';

        // Reset confidence bar visibility
        const confidenceContainer = document.querySelector('.confidence-container');
        confidenceContainer.style.display = '';

        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    function displayResult(data) {
        diseaseResult.textContent = data.disease;
        const confidencePercent = Math.round((data.confidence || 0) * 100);

        // Reset confidence container
        const confidenceContainer = document.querySelector('.confidence-container');
        confidenceContainer.style.display = '';
        confidenceBar.style.width = '0%';
        confidenceText.textContent = '0%';

        resultSection.classList.remove('hidden');

        // Animate progress bar after a short delay
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePercent}%`;

            let current = 0;
            const duration = 1000;
            if (confidencePercent === 0) {
                confidenceText.textContent = '0%';
                return;
            }
            const step = duration / confidencePercent;
            const counter = setInterval(() => {
                current++;
                confidenceText.textContent = `${current}%`;
                if (current >= confidencePercent) clearInterval(counter);
            }, step);
        }, 100);

        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Check backend health on page load
    fetch('http://localhost:8000/health', { method: 'GET' })
        .then(r => r.json())
        .then(data => {
            if (!data.model_loaded) {
                console.warn('Backend is up but model failed to load. Check server logs.');
            }
        })
        .catch(() => {
            console.warn('Backend not reachable. Make sure start_all.py is running.');
        });
});
