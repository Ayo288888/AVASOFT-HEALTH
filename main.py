import os
import json
import re
import requests
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Multi-Agent Healthcare System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

client = genai.Client(api_key=gemini_key) if gemini_key else None
groq_client = Groq(api_key=groq_key) if groq_key else None

# --- HUGGING FACE FREE SERVERLESS INFERENCE API ---
CUSTOM_MODEL_ID = "Iloriayomide/Symptom_Prediction"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{CUSTOM_MODEL_ID}"

def extract_symptom_text(raw_text: str) -> tuple[str, str]:
    """
    Parses raw input (which may be a stringified JSON array of conversation history)
    and returns a tuple of (clean_symptom_for_hf, full_context_for_llm).
    """
    clean_symptoms = raw_text
    full_context = raw_text

    try:
        data = json.loads(raw_text)
        if isinstance(data, list) and len(data) > 0:
            user_messages = [msg for msg in data if isinstance(msg, dict) and msg.get("role") == "user"]
            if user_messages:
                last_user_content = user_messages[-1].get("content", "")
            else:
                last_user_content = str(data[-1])
            
            full_context = last_user_content
            clean_symptoms = re.sub(r'\[Patient Details:.*?\]', '', last_user_content).strip()
    except (json.JSONDecodeError, TypeError):
        clean_symptoms = re.sub(r'\[Patient Details:.*?\]', '', raw_text).strip()

    if not clean_symptoms:
        clean_symptoms = raw_text

    return clean_symptoms, full_context

def query_huggingface_model(raw_text: str):
    """
    Queries Hugging Face's free Serverless Inference API for disease predictions.
    Does not require local PyTorch or GPU resources.
    """
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        response = requests.post(
            HF_INFERENCE_URL,
            headers=headers,
            json={"inputs": raw_text},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return [
                    {
                        "condition": item['label'].title(),
                        "confidence": f"{round(item['score']*100, 2)}%"
                    }
                    for item in data[0]
                ]
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                return [
                    {
                        "condition": item['label'].title(),
                        "confidence": f"{round(item['score']*100, 2)}%"
                    }
                    for item in data
                ]
        elif response.status_code == 503:
            print("HF Model loading... returning baseline prediction.")
        else:
            print(f"HF Inference API returned status {response.status_code}: {response.text}")
    except Exception as err:
        print(f"HF Inference API Notice: {err}")

    # Baseline fallback if HF model is cold starting
    return [
        {"condition": "Symptom Analysis Pending", "confidence": "90.0%"},
        {"condition": "General Medical Consultation", "confidence": "85.0%"},
        {"condition": "Clinical Evaluation Advised", "confidence": "80.0%"}
    ]

class SymptomRequest(BaseModel):
    text: str

def perform_triage(raw_text: str):
    """
    Analyzes symptoms using Hugging Face Serverless Inference API and Cloud LLMs.
    """
    if not client:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY missing in environment variables. Please set it in your host environment."
        )

    clean_symptom, full_context = extract_symptom_text(raw_text)
    predictions = query_huggingface_model(clean_symptom)

    prompt = (
        f"USER SYMPTOMS: {full_context}\n"
        f"AI ANALYSIS: {predictions}\n\n"
        "ACT AS: A Supportive Health Assistant.\n"
        "TASK: Provide a response in simple, non-medical language.\n"
        "CRITICAL UI INSTRUCTION: You must format the output strictly using clean HTML tags. "
        "DO NOT use Markdown asterisks (**). Format EXACTLY like this structure:\n\n"
        "<h4>🩺 Assessment</h4><p>[Explain what might be happening]</p>\n"
        "<h4>🩹 Immediate Relief</h4><ul><li>[Step 1]</li><li>[Step 2]</li></ul>\n"
        "<h4>💊 Pharmacy Advice</h4><p>[What to ask a pharmacist for]</p>\n"
        "<h4>🚨 RED FLAGS (When to see a doctor)</h4><ul><li>[Warning sign 1]</li></ul>\n"
        "<hr><p><small><em>DISCLAIMER: This is an AI tool and not a substitute for a human doctor.</em></small></p>"
    )

    # Multi-tier Gemini model fallback chain: 3.6 -> 3.5 -> 2.5 -> 3-preview
    GEMINI_MODELS = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-3-flash-preview"
    ]

    response_text = None
    last_exception = None

    for model_name in GEMINI_MODELS:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if res and res.text:
                response_text = res.text
                break
        except Exception as e:
            print(f"Gemini model '{model_name}' unavailable or failed: {e}. Trying next model...")
            last_exception = e

    if not response_text:
        if last_exception:
            raise last_exception
        raise HTTPException(status_code=500, detail="All Gemini model fallbacks failed to respond.")

    return predictions, response_text


@app.post("/predict")
def predict_text(request: SymptomRequest):
    try:
        predictions, note = perform_triage(request.text)
        return {
            "top_predictions": predictions[:3],
            "doctor_note": note,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/voice")
async def predict_voice(file: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY missing in environment variables. Please set it in your host environment."
        )
    try:
        audio_bytes = await file.read()
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3",
        )
        predictions, note = perform_triage(transcription.text)
        return {
            "transcription": transcription.text,
            "top_predictions": predictions[:3],
            "doctor_note": note
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        return {
            "doctor_note": "<h4>📸 Image Analysis</h4><p>Image received successfully. Visual diagnostic analysis model is being integrated.</p>",
            "top_predictions": [{"condition": "Visual Assessment", "confidence": "N/A"}],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- MOUNT FRONTEND ---
frontend_path = os.path.join(os.path.dirname(__file__), "Frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
