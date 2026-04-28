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

    predictBtn.addEventListener('click', async () => {
        const symptoms = symptomInput.value.trim();
        
        if (!symptoms) {
            alert('Please enter some symptoms first.');
            return;
        }

        // Show loader
        loader.classList.remove('hidden');
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symptoms }),
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const data = await response.json();
            
            // Hide loader and show result
            loader.classList.add('hidden');
            displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            loader.classList.add('hidden');
            alert('Failed to connect to the backend. Make sure the FastAPI server is running.');
        }
    });

    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        symptomInput.value = '';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    function displayResult(data) {
        diseaseResult.textContent = data.disease;
        const confidencePercent = Math.round(data.confidence * 100);
        
        resultSection.classList.remove('hidden');
        
        // Animate progress bar
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePercent}%`;
            
            // Count up confidence text
            let current = 0;
            const duration = 1000;
            const step = duration / confidencePercent;
            
            const counter = setInterval(() => {
                current++;
                confidenceText.textContent = `${current}%`;
                if (current >= confidencePercent) clearInterval(counter);
            }, step);
        }, 100);

        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
});
