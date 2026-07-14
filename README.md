AI Multi-Agent Healthcare System (Healthcare Chatbot)
This project is an AI-powered health assistant that analyzes user-described symptoms to predict potential diseases and gives medical advice and sugestions. it  uses multi-modal FastAPI triage engine that combines a custom Hugging Face classification model developed by Ilori Ayomide and a custom CNN model developed by Adejare Adelugba to deliver instant, voice-enabled patient care via a lightweight web UI.

## 🚀 Key Features

* **Multi-Agent Orchestration:** Uses a custom Hugging Face neural network developed by me (`Iloriayomide/Symptom_Prediction`) for rapid baseline classification, then passes the tensor outputs to Google's Gemini for natural language clinical reasoning.
* **Voice-Activated Triage:** Integrates Groq's ultra-fast `whisper-large-v3` model, allowing patients to simply speak their symptoms instead of typing.
* * **Image Upload ** Integrates Adejare's custom model, allowing patients to upload images for futher diagnosis.
  * Top-K Predictions: Returns the top 3 most likely disease matches with confidence percentages.
  * Natural Language Processing: Accepts full-sentence descriptions of symptoms (e.g., "I have a high fever and severe headache").

## 🧠 The Tech Stack
* **Backend:** FastAPI, Python, Uvicorn (Asynchronous REST API)
* **Local Machine Learning:** Hugging Face `transformers`, PyTorch
* **Cloud Intelligence:** Google GenAI SDK (Gemini 3 / 2.5), Groq Cloud (Whisper v3)
* **Security:** `python-dotenv` for strict environment variable isolation

🛠️ Tech Stack
Python 3.11+

Backend: FastAPI

ML Engine: Hugging Face Transformers & PyTorch

Server: Uvicorn

Frontend: HTML5, CSS3, JavaScript (Fetch API)

📦 Installation & Setup
1. Clone the Repository
Bash

git clone <your-repo-url>
cd healthcare-chatbot
2. Install Dependencies
Install the required Python packages listed in requirements.txt:

Bash

pip install -r requirements.txt
Dependencies include: fastapi, uvicorn, transformers, torch, and pydantic.

3. Run the Backend Server
Start the FastAPI server using Uvicorn. This will download the model the first time you run it.

Bash

uvicorn main:app --reload
The API will start at http://127.0.0.1:8000.

The model Iloriayomide/my-symptom-checker-biobert will be cached locally.

4. Launch the Frontend
Simply open the index.html file in your web browser.

You can double-click the file in your file explorer.

The frontend is pre-configured to send requests to http://127.0.0.1:8000/predict.

🚀 Usage
Ensure the backend terminal shows Application startup complete.

Open index.html in your browser.

Type your symptoms into the text area (e.g., "I am feeling dizzy and have a stomach ache").

Click "Analyze Symptoms".

View the top 3 predicted conditions and their confidence scores.


⚠️ Important Disclaimer
This is an Artificial Intelligence project for educational and demonstration purposes only.

It is not a doctor and should not be used for medical diagnosis.

The predictions are based on statistical patterns in text and may be inaccurate.

Always consult a certified medical professional for health advice.
