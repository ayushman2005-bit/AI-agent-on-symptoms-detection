# Health AI Agent with BioBERT Dashboard

This project is an AI-powered disease prediction agent that uses BioBERT to analyze symptoms and predict possible diseases. It features a modern, animated dashboard for a professional user experience.

## Features
- **BioBERT Model**: Uses `dmis-lab/biobert-base-cased-v1.1` for high-accuracy biomedical text understanding.
- **FastAPI Backend**: High-performance Python backend for real-time predictions.
- **Dynamic Dashboard**: Modern HTML5/CSS3 frontend with smooth animations and responsive design.
- **Data-Driven**: Trained on clinical symptom-disease association datasets.

## Project Structure
- `backend/`: FastAPI application and BioBERT model logic.
- `frontend/`: Dashboard files (HTML, CSS, JS).
- `run.py`: One-click script to start the entire application.

## How to Run in VSCode

1. **Open the Project**:
   Open the `health-ai-agent` folder in VSCode.

2. **Setup Environment** (Recommended):
   Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Train the Model** (Optional):
   The backend includes a `trainer.py` script. If you want to fine-tune the model on the provided dataset:
   ```bash
   cd backend
   python trainer.py
   cd ..
   ```
   *Note: This requires a GPU for efficient training.*

5. **Start the Application**:
   Run the convenience script:
   ```bash
   python start_all.py
   ```
   This will start the FastAPI server and open the dashboard in your default browser.

## Usage
1. Enter your symptoms in the text area (e.g., "itching, skin rash, fatigue").
2. Click **Analyze Symptoms**.
3. View the predicted disease and the AI's confidence level.

## Disclaimer
This application is for educational and demonstration purposes only. Always consult a qualified healthcare provider for medical diagnosis and treatment.
