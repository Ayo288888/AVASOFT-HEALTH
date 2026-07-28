import gradio as gr
from main import app

# Create a clean Gradio interface wrapper for Hugging Face Spaces (Gradio SDK Free Tier)
demo = gr.Blocks(title="AI Multi-Agent Healthcare Assistant")

with demo:
    gr.Markdown("""
    # 🩺 AI Multi-Agent Healthcare Assistant
    Welcome! The full web UI is hosted live on this space.
    - **Web Application:** Access the interactive UI at the root path (`/`).
    - **API Endpoints:** `/predict`, `/predict/voice`, `/predict/image`.
    """)

# Mount Gradio onto the existing FastAPI application
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
