import os
import json
import re
import time
import logging
import requests
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

# --- LOGGING CONFIG ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="AI Multi-Agent Healthcare System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        f"HTTP {request.method} {request.url.path} - Status: {response.status_code} - "
        f"Duration: {duration_ms}ms - Client IP: {client_ip}"
    )
    return response

# --- CONFIG ---
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    hf_token = hf_token.strip().strip('"').strip("'")

if not hf_token:
    logger.warning("HF_TOKEN environment variable is missing. Hugging Face requests may be unauthorized or rate limited.")

client = genai.Client(api_key=gemini_key) if gemini_key else None
groq_client = Groq(api_key=groq_key) if groq_key else None

# --- HUGGING FACE FREE SERVERLESS INFERENCE API ---
CUSTOM_MODEL_ID = "Iloriayomide/Symptom_Prediction"
HF_ENDPOINTS = [
    f"https://router.huggingface.co/hf-inference/v1/models/{CUSTOM_MODEL_ID}",
    f"https://router.huggingface.co/hf-inference/models/{CUSTOM_MODEL_ID}"
]

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
    Tests multiple endpoints and handles cold starts (HTTP 503) with retry logic.
    """
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    else:
        logger.warning("HF_TOKEN environment variable missing during Hugging Face query.")

    payload = {"inputs": raw_text}

    for url in HF_ENDPOINTS:
        for attempt in range(1, 4):
            logger.info(f"Querying Hugging Face endpoint (attempt {attempt}/3): {url}")
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=15
                )
                logger.info(f"HF endpoint {url} returned HTTP status code: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    predictions = []
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        predictions = [
                            {
                                "condition": item['label'].title(),
                                "confidence": f"{round(item['score']*100, 2)}%"
                            }
                            for item in data[0] if isinstance(item, dict) and 'label' in item and 'score' in item
                        ]
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        predictions = [
                            {
                                "condition": item['label'].title(),
                                "confidence": f"{round(item['score']*100, 2)}%"
                            }
                            for item in data if isinstance(item, dict) and 'label' in item and 'score' in item
                        ]

                    if predictions:
                        logger.info(f"HF Inference API request succeeded on {url}. Predictions: {predictions}")
                        return predictions
                    else:
                        logger.warning(f"HF Inference API response format unexpected from {url}: {data}")

                elif response.status_code == 401:
                    logger.error(
                        f"HF Inference API returned 401 Unauthorized for {url}: {response.text}. "
                        "HF_TOKEN in Render Environment Variables is invalid or lacks Serverless Inference permissions on Hugging Face. "
                        "To resolve this, generate a valid User Access Token at https://huggingface.co/settings/tokens with 'Make calls to the Serverless Inference API' permission."
                    )
                    break
                elif response.status_code == 503:
                    logger.warning(f"HF Model cold start / loading (503) on attempt {attempt}/3 for {url}. Response text: {response.text}")
                    if attempt < 3:
                        time.sleep(2)
                        continue
                else:
                    logger.error(f"HF Inference API returned non-200 status {response.status_code} for {url}: {response.text}")
                    break
            except Exception as err:
                logger.error(f"HF Inference API request error on {url} (attempt {attempt}/3): {err}")
                if attempt < 3:
                    time.sleep(1)

    logger.warning("All Hugging Face endpoints failed or timed out. Returning baseline fallback predictions.")
    # Baseline fallback if HF model is cold starting / unavailable
    return [
        {"condition": "Symptom Analysis Pending", "confidence": "90.0%"},
        {"condition": "General Medical Consultation", "confidence": "85.0%"},
        {"condition": "Clinical Evaluation Advised", "confidence": "80.0%"}
    ]

class SymptomRequest(BaseModel):
    text: str

def parse_llm_json_response(text: str) -> dict | None:
    """
    Parses LLM response text into a JSON dictionary, stripping markdown formatting if needed.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Failed to parse LLM JSON response directly: {e}")
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return None


def generate_llm_response(prompt: str) -> str:
    """
    Executes LLM generation with multi-tier Gemini fallback chain and optional Groq backup.
    """
    GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview"
    ]
    last_exception = None

    if client:
        for model_name in GEMINI_MODELS:
            logger.info(f"Attempting Gemini generation with model: '{model_name}'")
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if res and res.text:
                    logger.info(f"Gemini generation succeeded with model: '{model_name}'")
                    return res.text
            except Exception as e:
                logger.warning(f"Gemini model '{model_name}' unavailable or failed: {e}. Trying next model...")
                last_exception = e

    if groq_client:
        logger.info("Attempting fallback generation with Groq LLM...")
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            if chat_completion.choices and chat_completion.choices[0].message.content:
                logger.info("Groq LLM generation succeeded.")
                return chat_completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq LLM generation failed: {e}")
            last_exception = e

    if last_exception:
        raise last_exception
    raise HTTPException(
        status_code=500,
        detail="LLM generation failed across all available Gemini and Groq models."
    )


def perform_triage(raw_text: str):
    """
    Analyzes symptoms using Hugging Face Serverless Inference API and Cloud LLMs.
    If HF model fails or returns pending fallback, dynamically classifies diseases via Gemini/Groq LLM.
    """
    if not client and not groq_client:
        logger.error("Neither GEMINI_API_KEY nor GROQ_API_KEY configured in environment variables.")
        raise HTTPException(
            status_code=500,
            detail="LLM API key missing in environment variables. Please set GEMINI_API_KEY or GROQ_API_KEY."
        )

    clean_symptom, full_context = extract_symptom_text(raw_text)
    predictions = query_huggingface_model(clean_symptom)

    is_fallback = (
        not predictions
        or not isinstance(predictions, list)
        or len(predictions) == 0
        or predictions[0].get("condition") == "Symptom Analysis Pending"
    )

    if is_fallback:
        logger.info("HF prediction unavailable or pending. Triggering dynamic Gemini AI disease classification and triage fallback.")
        fallback_prompt = (
            f"USER SYMPTOMS: {full_context}\n\n"
            "ACT AS: An Expert AI Medical Triage Specialist.\n"
            "TASK: Perform structured disease classification and generate a clinical triage doctor note based on the patient's symptoms.\n"
            "You MUST respond ONLY with a single valid JSON object formatted strictly as follows (no markdown outside the JSON):\n"
            "{\n"
            '  "top_predictions": [\n'
            '    {"condition": "Primary Condition", "confidence": "85.0%"},\n'
            '    {"condition": "Secondary Condition", "confidence": "70.0%"},\n'
            '    {"condition": "Differential Diagnosis", "confidence": "55.0%"}\n'
            "  ],\n"
            '  "doctor_note": "<h4>🩺 Assessment</h4><p>[Explain what might be happening based on symptoms]</p><h4>🩹 Immediate Relief</h4><ul><li>[Step 1]</li><li>[Step 2]</li></ul><h4>💊 Pharmacy Advice</h4><p>[What to ask a pharmacist for]</p><h4>🚨 RED FLAGS (When to see a doctor)</h4><ul><li>[Warning sign 1]</li></ul><hr><p><small><em>DISCLAIMER: This is an AI tool and not a substitute for a human doctor.</em></small></p>"\n'
            "}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. 'top_predictions' must be an array of top 3 suspected medical conditions with realistic confidence percentages based on the patient's symptoms (formatted as 'XX.X%').\n"
            "2. 'doctor_note' must be clean HTML formatted response with Assessment, Immediate Relief, Pharmacy Advice, Red Flags, and Disclaimer. DO NOT use Markdown asterisks (**).\n"
        )

        response_text = generate_llm_response(fallback_prompt)
        parsed = parse_llm_json_response(response_text)

        if parsed and isinstance(parsed, dict) and "top_predictions" in parsed and "doctor_note" in parsed:
            llm_predictions = parsed.get("top_predictions", [])
            doctor_note = parsed.get("doctor_note", "")
            if isinstance(llm_predictions, list) and len(llm_predictions) > 0:
                return llm_predictions, doctor_note

        doctor_note = response_text
        if parsed and isinstance(parsed, dict) and "doctor_note" in parsed:
            doctor_note = parsed["doctor_note"]

        predictions = [
            {"condition": "Clinical Symptom Evaluation", "confidence": "85.0%"},
            {"condition": "General Medical Assessment", "confidence": "75.0%"},
            {"condition": "Differential Diagnosis Pending", "confidence": "65.0%"}
        ]
        return predictions, doctor_note

    else:
        logger.info("HF prediction succeeded. Generating LLM doctor note.")
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

        response_text = generate_llm_response(prompt)
        return predictions, response_text


@app.post("/predict")
def predict_text(request: SymptomRequest):
    logger.info("Received text prediction request.")
    try:
        predictions, note = perform_triage(request.text)
        return {
            "top_predictions": predictions[:3],
            "doctor_note": note,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing text prediction request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/voice")
async def predict_voice(file: UploadFile = File(...)):
    logger.info(f"Received voice prediction request. File: {file.filename}, Content-Type: {file.content_type}")
    if not groq_client:
        logger.error("GROQ_API_KEY missing in environment variables.")
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY missing in environment variables. Please set it in your host environment."
        )
    try:
        audio_bytes = await file.read()
        logger.info("Transcribing voice audio using Groq Whisper model...")
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3",
        )
        logger.info(f"Voice transcription successful: '{transcription.text[:60]}...'")
        predictions, note = perform_triage(transcription.text)
        return {
            "transcription": transcription.text,
            "top_predictions": predictions[:3],
            "doctor_note": note
        }
    except Exception as e:
        logger.error(f"Error processing voice prediction request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    logger.info(f"Received image prediction request. File: {file.filename}, Content-Type: {file.content_type}")
    try:
        return {
            "doctor_note": "<h4>📸 Image Analysis</h4><p>Image received successfully. Visual diagnostic analysis model is being integrated.</p>",
            "top_predictions": [{"condition": "Visual Assessment", "confidence": "N/A"}],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing image prediction request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- MOUNT FRONTEND ---
frontend_path = os.path.join(os.path.dirname(__file__), "Frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)