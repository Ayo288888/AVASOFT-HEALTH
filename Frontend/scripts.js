/* ========== SYSTEM STATE & CONTEXT ========== */
let conversationHistory = []; 
let uploadedImages = [];
let isRecording = false;
let mediaRecorder;
let audioChunks = [];

/* ========== 1. VOICE TRIAGE (REPAIRED FOR NEW UI) ========== */
async function toggleRecording() {
    const micIcon = document.getElementById('micIcon');
    const micBtn = document.getElementById('micBtn');

    if (!isRecording) {
        console.log("Starting voice recording...");
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                await sendVoiceData(audioBlob);
            };

            mediaRecorder.start();
            isRecording = true;
            micIcon.textContent = '🛑';
            micBtn.classList.add('recording-pulse'); // Add a CSS pulse if you have one
            showNotification('Recording... Speak now.', 'info');
        } catch (err) {
            console.error("Mic Error:", err);
            alert('Microphone access denied. Please enable it in browser settings.');
        }
    } else {
        console.log("Stopping voice recording...");
        mediaRecorder.stop();
        isRecording = false;
        micIcon.textContent = '🎤';
        micBtn.classList.remove('recording-pulse');
    }
}

async function sendVoiceData(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'audio.wav');
    
    // Add a temporary "Processing" bubble
    appendMessage("ai", "Transcribing your voice... 🎙️");

    try {
        const response = await fetch('/predict/voice', { method: 'POST', body: formData });
        const data = await response.json();
        
        // Remove the "Processing" bubble (optional, or just append the next)
        document.getElementById("symptoms").value = data.transcription;
        
        // Automatically run the analysis
        initiateScan();
    } catch (e) {
        console.error("Transcription Error:", e);
        appendMessage("ai", "⚠️ Voice engine failed. Please try typing your symptoms.");
    }
}

/* ========== 2. CORE MESSAGE HANDLER ========== */
function appendMessage(role, text, predictions = null) {
    const thread = document.getElementById("chatThread");
    const msgDiv = document.createElement("div");
    msgDiv.className = role === "user" ? "user-message" : "ai-message";
    
    // Formatting logic
    const formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    let diagHTML = '';
    if (predictions && predictions.length > 0) {
        diagHTML = `
            <div class="differential-box" style="margin-top:15px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
                <p style="font-size:0.7rem; opacity:0.6; text-transform:uppercase;">Differential Diagnosis</p>
                ${predictions.map(p => `
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-top:5px;">
                        <span>${p.disease || p.condition}</span>
                        <span style="color:var(--primary); font-weight:700;">${p.confidence}</span>
                    </div>
                `).join('')}
            </div>`;
    }

    msgDiv.innerHTML = `
        <div class="message-icon">${role === 'user' ? '👤' : '🤖'}</div>
        <div class="message-content">${formattedText}${diagHTML}</div>
    `;
    
    thread.appendChild(msgDiv);
    thread.scrollTop = thread.scrollHeight;
}

/* ========== 3. ANALYSIS LOGIC ========== */
async function initiateScan() {
    const textarea = document.getElementById("symptoms");
    const textInput = textarea.value.trim();
    
    if (!textInput && uploadedImages.length === 0) return;

    appendMessage("user", textInput || "Analyzing image...");
    textarea.value = "";
    textarea.rows = 1;

    // Show "AI is thinking" placeholder
    const thinkingId = "thinking-" + Date.now();
    const thread = document.getElementById("chatThread");
    const thinkingDiv = document.createElement("div");
    thinkingDiv.className = "ai-message";
    thinkingDiv.id = thinkingId;
    thinkingDiv.innerHTML = `<div class="message-icon">🤖</div><div class="message-content">Analyzing clinical data... <span class="loading-dots">...</span></div>`;
    thread.appendChild(thinkingDiv);

    try {
        let response;
        if (uploadedImages.length > 0) {
            const formData = new FormData();
            formData.append('file', uploadedImages[0].file);
            response = await fetch('/predict/image', { method: 'POST', body: formData });
            uploadedImages = [];
            document.getElementById('imagePreview').innerHTML = "";
        } else {
            conversationHistory.push({ role: "user", content: textInput });
            response = await fetch('/predict', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: JSON.stringify(conversationHistory) }) 
            });
        }

        const data = await response.json();
        document.getElementById(thinkingId).remove(); // Remove thinking placeholder
        
        const predictions = data.top_predictions || data.analysis || data.symptom_analysis;
        appendMessage("ai", data.doctor_note, predictions);
        conversationHistory.push({ role: "assistant", content: data.doctor_note });

    } catch (error) {
        document.getElementById(thinkingId).innerHTML = `<div class="message-icon">🤖</div><div class="message-content">⚠️ System error: Engine unreachable.</div>`;
    }
}

/* ========== 4. UTILS & CANVAS ========== */
function onUserType() {
    const textarea = document.getElementById("symptoms");
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        uploadedImages = [{ file: file, url: e.target.result }];
        document.getElementById('imagePreview').innerHTML = `<div class="mini-img"><img src="${e.target.result}"><span onclick="uploadedImages=[]; document.getElementById('imagePreview').innerHTML=''">×</span></div>`;
    };
    reader.readAsDataURL(file);
}

function clearChat() {
    document.getElementById("chatThread").innerHTML = "";
    conversationHistory = [];
    appendMessage("ai", "Session reset. How can I assist you today?");
}

// Neural Background
const canvas = document.getElementById("neuralCanvas");
const ctx = canvas.getContext("2d");
let particles = [];
function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
window.addEventListener("resize", resize); resize();
class Particle {
    constructor() { this.x = Math.random()*canvas.width; this.y = Math.random()*canvas.height; this.vx = (Math.random()-0.5)*0.5; this.vy = (Math.random()-0.5)*0.5; }
    update() { this.x += this.vx; this.y += this.vy; if (this.x<0 || this.x>canvas.width) this.vx*=-1; if (this.y<0 || this.y>canvas.height) this.vy*=-1; }
    draw() { ctx.beginPath(); ctx.arc(this.x, this.y, 1, 0, Math.PI*2); ctx.fillStyle = "rgba(99, 102, 241, 0.2)"; ctx.fill(); }
}
for(let i=0; i<60; i++) particles.push(new Particle());
function animate() {
    ctx.fillStyle = 'rgba(2, 6, 23, 0.05)'; ctx.fillRect(0,0,canvas.width,canvas.height);
    particles.forEach(p => { p.update(); p.draw(); });
    requestAnimationFrame(animate);
}
animate();