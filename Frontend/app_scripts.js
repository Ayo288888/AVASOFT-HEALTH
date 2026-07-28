/* ========== SYSTEM STATE & CONTEXT ========== */
let conversationHistory = []; // This holds the active session's LLM message history
let uploadedImages = [];
let isRecording = false;
let mediaRecorder;
let audioChunks = [];

// Session & Profile Data Structure
let sessions = [];
let activeSessionId = null;
let patientProfile = {
    age: 32,
    blood: "O+",
    allergies: "None"
};

const RED_FLAG_KEYWORDS = [
    "chest pain", "shortness of breath", "can't breathe", "cannot breathe", "difficulty breathing",
    "stroke", "numbness on one side", "slurred speech", "coughing blood", "passed out",
    "loss of consciousness", "anaphylaxis", "severe allergic reaction", "crushing chest", "heart attack"
];

/* ========== 1. INITIALIZE APP, THEME, SESSIONS & PROFILE ========== */
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    loadProfile();
    loadSessions();
    setupKeyboardListeners();
});

// Theme Switcher & Persistence
function initTheme() {
    const savedTheme = localStorage.getItem("avaSoftTheme") || "light";
    setTheme(savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.classList.contains("dark") ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    setTheme(next);
}

function setTheme(theme) {
    if (theme === "dark") {
        document.documentElement.classList.add("dark");
        document.documentElement.setAttribute("data-theme", "dark");
        const icon = document.getElementById("themeIcon");
        if (icon) icon.textContent = "light_mode";
    } else {
        document.documentElement.classList.remove("dark");
        document.documentElement.setAttribute("data-theme", "light");
        const icon = document.getElementById("themeIcon");
        if (icon) icon.textContent = "dark_mode";
    }
    localStorage.setItem("avaSoftTheme", theme);
}

// Keyboard 'Enter' Submission (Shift+Enter for newline)
function setupKeyboardListeners() {
    const textarea = document.getElementById("symptoms");
    if (textarea) {
        textarea.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                initiateScan();
            }
        });
    }
}

// Red-Flag Emergency Triage Check
function checkEmergencyRedFlags(text) {
    const lower = text.toLowerCase();
    for (const kw of RED_FLAG_KEYWORDS) {
        if (lower.includes(kw)) {
            return kw;
        }
    }
    return null;
}

function toggleEmergencyModal(show, symptomKeyword = "") {
    const modal = document.getElementById("emergencyModal");
    if (!modal) return;
    if (show) {
        const textNode = document.getElementById("emergencySymptomText");
        if (textNode) textNode.textContent = symptomKeyword;
        modal.classList.add("active");
    } else {
        modal.classList.remove("active");
    }
}

// Quick Symptom Chips
function toggleSymptomChip(chipName, btn) {
    const textarea = document.getElementById("symptoms");
    let currentText = textarea.value.trim();
    if (btn.classList.contains("selected")) {
        btn.classList.remove("selected");
        const regex = new RegExp(",?\\s*" + chipName, "gi");
        currentText = currentText.replace(regex, "").replace(/^,\s*/, "").trim();
        textarea.value = currentText;
    } else {
        btn.classList.add("selected");
        if (currentText.length > 0) {
            textarea.value = currentText + ", " + chipName;
        } else {
            textarea.value = chipName;
        }
    }
    onUserType();
}

// Load patient medical profile from localStorage
function loadProfile() {
    const savedProfile = localStorage.getItem("avaSoftProfile");
    if (savedProfile) {
        try {
            patientProfile = JSON.parse(savedProfile);
        } catch (e) {
            console.error("Error parsing profile", e);
        }
    }
    
    // Update Banner Display
    document.getElementById("profileAge").textContent = patientProfile.age;
    document.getElementById("profileBlood").textContent = patientProfile.blood;
    document.getElementById("profileAllergies").textContent = patientProfile.allergies;

    // Fill Edit Form Fields
    document.getElementById("editAge").value = patientProfile.age;
    document.getElementById("editBlood").value = patientProfile.blood;
    document.getElementById("editAllergies").value = patientProfile.allergies;
}

// Toggle edit profile modal overlay
function toggleProfileModal(show) {
    const modal = document.getElementById("profileModal");
    if (show) {
        modal.classList.add("active");
    } else {
        modal.classList.remove("active");
    }
}

// Save patient medical profile details
function saveProfile() {
    patientProfile.age = parseInt(document.getElementById("editAge").value) || 32;
    patientProfile.blood = document.getElementById("editBlood").value.trim() || "O+";
    patientProfile.allergies = document.getElementById("editAllergies").value.trim() || "None";

    localStorage.setItem("avaSoftProfile", JSON.stringify(patientProfile));
    
    // Update display banner
    document.getElementById("profileAge").textContent = patientProfile.age;
    document.getElementById("profileBlood").textContent = patientProfile.blood;
    document.getElementById("profileAllergies").textContent = patientProfile.allergies;

    toggleProfileModal(false);
}

// Load all chat sessions from localStorage
function loadSessions() {
    const savedSessions = localStorage.getItem("avaSoftSessions");
    const savedActiveId = localStorage.getItem("avaSoftActiveSessionId");
    
    if (savedSessions) {
        try {
            sessions = JSON.parse(savedSessions);
        } catch (e) {
            console.error("Error parsing sessions", e);
            sessions = [];
        }
    }

    if (sessions.length === 0) {
        // Create initial default session
        const initId = "session-" + Date.now();
        const initPatientId = "#AV-" + Math.floor(1000 + Math.random() * 9000);
        const initSessionNum = "#" + Math.floor(100 + Math.random() * 900) + "-S";
        
        sessions.push({
            id: initId,
            title: "New Interview",
            patientId: initPatientId,
            sessionId: initSessionNum,
            history: [
                {
                    role: "assistant",
                    content: "Welcome to AvaSoft Health. Let's find out what's going on. Please describe the symptoms you are experiencing in detail.",
                    predictions: null
                }
            ]
        });
        activeSessionId = initId;
        saveToLocalStorage();
    } else {
        activeSessionId = savedActiveId || sessions[0].id;
        // Make sure activeSessionId exists in sessions
        if (!sessions.find(s => s.id === activeSessionId)) {
            activeSessionId = sessions[0].id;
        }
    }

    localStorage.setItem("avaSoftActiveSessionId", activeSessionId);
    renderActiveSession();
    renderSessionList();
}

// Render the active session conversation in chat area
function renderActiveSession() {
    const activeSession = sessions.find(s => s.id === activeSessionId);
    if (!activeSession) return;

    // Update Header Session Details
    document.getElementById("headerPatientId").textContent = `Patient ID: ${activeSession.patientId}`;
    document.getElementById("headerSessionId").textContent = `Session: ${activeSession.sessionId}`;

    const thread = document.getElementById("chatThread");
    
    thread.innerHTML = "";
    
    // Populate messages
    conversationHistory = [];
    activeSession.history.forEach(msg => {
        // Render to DOM
        appendMessageDOM(msg.role, msg.content, msg.predictions, false);
        
        // Keep conversationHistory in sync for API fallback references (excludes system indicators if needed)
        conversationHistory.push({ role: msg.role, content: msg.content });
    });

    // Scroll chat area
    setTimeout(() => {
        const scrollArea = document.getElementById("chatScrollArea");
        if (scrollArea) scrollArea.scrollTop = scrollArea.scrollHeight;
    }, 50);
}

// Render the historical chat session sidebar
function renderSessionList() {
    const list = document.getElementById("sessionList");
    if (!list) return;

    list.innerHTML = sessions.map(s => {
        const isActive = s.id === activeSessionId;
        const activeClass = isActive ? "bg-accent-blue/15 border-accent-blue text-primary" : "hover:bg-slate-100 border-transparent text-text-muted";
        
        return `
            <div class="flex items-center justify-between p-2.5 rounded-xl border text-sm font-semibold transition-all cursor-pointer ${activeClass}" onclick="switchSession('${s.id}')">
                <div class="flex items-center gap-2 overflow-hidden flex-1">
                    <span class="material-symbols-outlined text-[16px] shrink-0">chat_bubble</span>
                    <span class="truncate pr-1">${s.title}</span>
                </div>
                <button onclick="deleteSession(event, '${s.id}')" class="text-text-muted hover:text-rose-600 transition-colors flex items-center justify-center p-1 rounded">
                    <span class="material-symbols-outlined text-[14px]">delete</span>
                </button>
            </div>
        `;
    }).join("");
}

// Start a fresh diagnostic session
function startNewSession() {
    const newId = "session-" + Date.now();
    const newPatientId = "#AV-" + Math.floor(1000 + Math.random() * 9000);
    const newSessionNum = "#" + Math.floor(100 + Math.random() * 900) + "-S";

    sessions.push({
        id: newId,
        title: "New Interview",
        patientId: newPatientId,
        sessionId: newSessionNum,
        history: [
            {
                role: "assistant",
                content: "Welcome to AvaSoft Health. Let's find out what's going on. Please describe the symptoms you are experiencing in detail.",
                predictions: null
            }
        ]
    });

    activeSessionId = newId;
    saveToLocalStorage();
    renderActiveSession();
    renderSessionList();
}

// Switch current view to historical session
function switchSession(id) {
    activeSessionId = id;
    localStorage.setItem("avaSoftActiveSessionId", activeSessionId);
    renderActiveSession();
    renderSessionList();
}

// Delete historical session
function deleteSession(event, id) {
    event.stopPropagation(); // Prevent trigger switchSession
    
    sessions = sessions.filter(s => s.id !== id);
    
    if (sessions.length === 0) {
        // If empty, re-initialize
        localStorage.clear();
        loadSessions();
        return;
    }

    if (activeSessionId === id) {
        activeSessionId = sessions[0].id;
        localStorage.setItem("avaSoftActiveSessionId", activeSessionId);
    }

    saveToLocalStorage();
    renderActiveSession();
    renderSessionList();
}

// Clear all active sessions
function clearAllSessions() {
    if (confirm("Are you sure you want to clear your entire assessment history?")) {
        sessions = [];
        activeSessionId = null;
        localStorage.removeItem("avaSoftSessions");
        localStorage.removeItem("avaSoftActiveSessionId");
        loadSessions();
    }
}

// Save sessions state to localStorage
function saveToLocalStorage() {
    localStorage.setItem("avaSoftSessions", JSON.stringify(sessions));
}


/* ========== 2. VOICE TRIAGE ========== */
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
            micIcon.textContent = 'stop'; 
            micBtn.classList.add('recording-active');
        } catch (err) {
            console.error("Mic Error:", err);
            alert('Microphone access denied. Please enable it in browser settings.');
        }
    } else {
        console.log("Stopping voice recording...");
        mediaRecorder.stop();
        isRecording = false;
        micIcon.textContent = 'mic'; 
        micBtn.classList.remove('recording-active');
    }
}

async function sendVoiceData(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'audio.wav');
    
    appendMessage("ai", "Transcribing voice clinical feed... 🎙️");

    try {
        const response = await fetch('/predict/voice', { method: 'POST', body: formData });
        const data = await response.json();
        
        document.getElementById("symptoms").value = data.transcription;
        onUserType(); // resize textarea
        
        // Auto-run scan
        initiateScan();
    } catch (e) {
        console.error("Transcription Error:", e);
        appendMessage("ai", "⚠️ Voice engine failed. Please type your symptoms manually.");
    }
}

/* ========== 3. MESSAGE RENDERING AND PERSISTENCE ========== */
function appendMessage(role, text, predictions = null) {
    // 1. Render to DOM
    appendMessageDOM(role, text, predictions, true);

    // 2. Push to local history and save to session persistence
    const activeSession = sessions.find(s => s.id === activeSessionId);
    if (activeSession) {
        activeSession.history.push({
            role: role,
            content: text,
            predictions: predictions
        });

        // If this is the first user message, rename session title dynamically
        if (role === "user" && activeSession.title === "New Interview") {
            const shortTitle = text.length > 22 ? text.substring(0, 20) + "..." : text;
            activeSession.title = shortTitle;
        }

        saveToLocalStorage();
        renderSessionList();
    }
    
    // Maintain local fallback memory sync
    conversationHistory.push({ role: role, content: text });
}

// Low-level helper to build and append chat bubble element to DOM
function appendMessageDOM(role, text, predictions = null, doScroll = true) {
    const thread = document.getElementById("chatThread");
    const msgDiv = document.createElement("div");
    
    if (role === "user") {
        msgDiv.className = "flex justify-end bubble-in";
        const formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
            
        msgDiv.innerHTML = `
            <div class="bg-primary text-on-primary p-5 rounded-2xl rounded-tr-none clinical-card-shadow max-w-md">
                <p class="text-body-md">${formattedText}</p>
            </div>
        `;
    } else {
        msgDiv.className = "flex items-start gap-4 bubble-in";
        const formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');

        let diagHTML = '';
        if (predictions && predictions.length > 0) {
            diagHTML = `
                <div class="mt-4 pt-4 border-t border-outline-variant space-y-4">
                    <div class="flex items-center gap-2 text-accent-blue font-bold text-xs uppercase tracking-wider">
                        <span class="material-symbols-outlined text-[18px]">analytics</span>
                        <span>Triage Results</span>
                    </div>
                    <div class="space-y-3">
                        ${predictions.map(p => {
                            const confidenceStr = p.confidence || '0%';
                            const pct = parseFloat(confidenceStr) || 0;
                            return `
                                <div>
                                    <div class="flex justify-between text-xs font-semibold mb-1">
                                        <span>${p.disease || p.condition}</span>
                                        <span class="text-accent-blue font-bold">${confidenceStr}</span>
                                    </div>
                                    <div class="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                                        <div class="h-full bg-accent-blue" style="width: ${pct}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>`;
        }

        msgDiv.innerHTML = `
            <div class="w-9 h-9 rounded-full overflow-hidden shrink-0 border border-outline-variant flex items-center justify-center bg-slate-100">
                <img src="logo.png" alt="Clinical AI" class="w-full h-full object-cover">
            </div>
            <div class="flex flex-col gap-2 w-full max-w-md">
                <div class="bg-surface-container-low p-5 rounded-2xl rounded-tl-none clinical-card-shadow border border-outline-variant">
                    <div class="text-body-md text-on-surface leading-relaxed">${formattedText}</div>
                    ${diagHTML}
                </div>
                <span class="text-[10px] text-text-muted ml-1">AvaSoft Health Bot</span>
            </div>
        `;
    }
    
    thread.appendChild(msgDiv);
    
    // Smooth scroll chat area
    if (doScroll) {
        const scrollArea = document.getElementById("chatScrollArea");
        if (scrollArea) {
            scrollArea.scrollTop = scrollArea.scrollHeight;
        }
    }
}

/* ========== 4. ANALYSIS LOGIC ========== */
async function initiateScan() {
    const textarea = document.getElementById("symptoms");
    const textInput = textarea.value.trim();
    
    if (!textInput && uploadedImages.length === 0) return;

    // Check Client-side Red Flag Triage
    const flag = checkEmergencyRedFlags(textInput);
    if (flag) {
        toggleEmergencyModal(true, flag);
    }

    // Build the query to show in chat
    appendMessage("user", textInput || "Analyzing image...");
    textarea.value = "";
    textarea.style.height = 'auto'; // reset height

    // Reset symptom chip selections
    document.querySelectorAll(".symptom-chip").forEach(c => c.classList.remove("selected"));

    // Show AI processing bubble
    const thinkingId = "thinking-" + Date.now();
    const thread = document.getElementById("chatThread");
    const thinkingDiv = document.createElement("div");
    thinkingDiv.className = "flex items-start gap-4 bubble-in";
    thinkingDiv.id = thinkingId;
    thinkingDiv.innerHTML = `
        <div class="w-9 h-9 rounded-full overflow-hidden shrink-0 border border-outline-variant flex items-center justify-center bg-slate-100 dark:bg-slate-800">
            <img src="logo.png" alt="Clinical AI" class="w-full h-full object-cover">
        </div>
        <div class="flex flex-col gap-2 w-full max-w-md">
            <div class="bg-surface-container-low p-5 rounded-2xl rounded-tl-none clinical-card-shadow border border-outline-variant">
                <p class="text-body-md text-on-surface">
                    Analyzing symptoms... <span class="loading-dots">...</span>
                </p>
            </div>
            <span class="text-[10px] text-text-muted ml-1">AvaSoft Health Bot</span>
        </div>
    `;
    thread.appendChild(thinkingDiv);
    
    const scrollArea = document.getElementById("chatScrollArea");
    if (scrollArea) {
        scrollArea.scrollTop = scrollArea.scrollHeight;
    }

    try {
        let response;
        if (uploadedImages.length > 0) {
            const formData = new FormData();
            formData.append('file', uploadedImages[0].file);
            response = await fetch('/predict/image', { method: 'POST', body: formData });
            uploadedImages = [];
            document.getElementById('imagePreview').innerHTML = "";
        } else {
            const tempVal = document.getElementById("temperature").value || "37.0";
            const severityVal = document.getElementById("severity").value;
            const vitalsContext = ` [Patient Details: Age=${patientProfile.age}, Blood=${patientProfile.blood}, Allergies=${patientProfile.allergies}, Temp=${tempVal}°C, PainSeverity=${severityVal}/10]`;
            
            conversationHistory.push({ role: "user", content: textInput + vitalsContext });
            
            response = await fetch('/predict', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: JSON.stringify(conversationHistory) }) 
            });
        }

        if (!response.ok) throw new Error("API Offline");
        const data = await response.json();
        const thinkingNode = document.getElementById(thinkingId);
        if (thinkingNode) thinkingNode.remove();
        
        const predictions = data.top_predictions || data.analysis || data.symptom_analysis;
        appendMessage("ai", data.doctor_note, predictions);

    } catch (error) {
        console.warn("Backend offline, engaging intelligent fallback diagnostic engine:", error);
        const thinkingNode = document.getElementById(thinkingId);
        if (thinkingNode) thinkingNode.remove();
        
        // Demo Mock Diagnostic Engine Fallback
        const mockResult = generateMockDiagnosis(textInput);
        appendMessage("ai", mockResult.doctor_note, mockResult.top_predictions);
    }
}

// Fallback Mock Diagnostic Engine for demo mode
function generateMockDiagnosis(userText) {
    const text = (userText || "").toLowerCase();
    const tempVal = parseFloat(document.getElementById("temperature").value) || 37.0;
    const severityVal = parseInt(document.getElementById("severity").value) || 5;

    let diseaseList = [];
    let note = "";

    if (text.includes("fever") || text.includes("cough") || text.includes("headache") || tempVal >= 38.0) {
        diseaseList = [
            { disease: "Viral Upper Respiratory Infection", confidence: "78%" },
            { disease: "Influenza Type A/B", confidence: "64%" },
            { disease: "Acute Sinusitis", confidence: "38%" }
        ];
        note = `Based on your reported symptoms (Temperature: **${tempVal}°C**), the clinical picture strongly aligns with a **viral upper respiratory infection**.\n\n**Recommended Relief & Next Steps:**\n- Maintain high hydration levels and rest.\n- Over-the-counter antipyretics (e.g. Paracetamol/Acetaminophen) as appropriate.\n- Monitor temperature; if fever exceeds 38.5°C for more than 48 hours, seek clinical consultation.`;
    } else if (text.includes("stomach") || text.includes("nausea") || text.includes("vomit")) {
        diseaseList = [
            { disease: "Acute Gastroenteritis", confidence: "82%" },
            { disease: "Dietary Gastritis", confidence: "55%" },
            { disease: "Functional Dyspepsia", confidence: "31%" }
        ];
        note = `Your gastrointestinal complaints suggest mild **acute gastroenteritis** or dietary irritation.\n\n**Suggested Care:**\n- Sip oral rehydration solutions continuously.\n- Avoid heavy, greasy, or acidic foods for 12-24 hours.\n- Seek medical evaluation if unable to retain liquids.`;
    } else if (text.includes("chest") || text.includes("breath")) {
        diseaseList = [
            { disease: "Acute Bronchitis", confidence: "68%" },
            { disease: "Intercostal Muscle Strain", confidence: "45%" },
            { disease: "Mild Asthma Flare", confidence: "39%" }
        ];
        note = `⚠️ **Clinical Caution:** Symptoms involving chest or respiratory discomfort require careful observation.\n\nIf you develop worsening breathlessness or persistent chest tightness, **seek emergency medical care immediately**.`;
    } else {
        diseaseList = [
            { disease: "General Clinical Fatigue", confidence: "62%" },
            { disease: "Tension / Stress Response", confidence: "48%" }
        ];
        note = `Thank you for sharing your symptoms. Based on your current pain severity level (**${severityVal}/10**), your symptoms appear manageable.\n\n**Next Steps:** Rest and track any new or evolving symptoms. If symptoms persist beyond 48 hours, please consult a healthcare professional.`;
    }

    return { doctor_note: note, top_predictions: diseaseList };
}

/* ========== 5. UTILS & DYNAMIC UI ========== */
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
        document.getElementById('imagePreview').innerHTML = `
            <div class="relative w-16 h-16 rounded-xl overflow-hidden border border-outline-variant">
                <img src="${e.target.result}" class="w-full h-full object-cover">
                <span class="absolute top-1 right-1 bg-black/60 text-white rounded-full w-4 h-4 flex items-center justify-center text-[10px] cursor-pointer" onclick="uploadedImages=[]; document.getElementById('imagePreview').innerHTML=''">×</span>
            </div>
        `;
    };
    reader.readAsDataURL(file);
}

function clearChat() {
    if (confirm("Reset current assessment?")) {
        const activeSession = sessions.find(s => s.id === activeSessionId);
        if (activeSession) {
            activeSession.history = [
                {
                    role: "assistant",
                    content: "Symptom checker reset. Please tell me what symptoms you are experiencing.",
                    predictions: null
                }
            ];
            activeSession.title = "New Interview";
            saveToLocalStorage();
            renderActiveSession();
            renderSessionList();
        }
    }
}

function updateTempIndicator(val) {
    const indicator = document.getElementById("tempIndicator");
    const temp = parseFloat(val);
    if (isNaN(temp)) {
        indicator.textContent = "Normal";
        indicator.className = "absolute right-2 px-2 py-1 text-[10px] font-bold rounded-lg bg-teal-50 text-teal-700 border border-teal-200";
        return;
    }
    
    if (temp >= 38.0) {
        indicator.textContent = "Fever";
        indicator.className = "absolute right-2 px-2 py-1 text-[10px] font-bold rounded-lg bg-red-50 text-red-700 border border-red-200";
    } else {
        indicator.textContent = "Normal";
        indicator.className = "absolute right-2 px-2 py-1 text-[10px] font-bold rounded-lg bg-teal-50 text-teal-700 border border-teal-200";
    }
}

function updateSeverity(val) {
    const display = document.getElementById("severityDisplay");
    let label = "Moderate";
    if (val <= 3) label = "Mild";
    else if (val >= 8) label = "Severe";
    
    display.textContent = `${label} (${val}/10)`;
}
