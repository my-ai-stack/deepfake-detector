#!/usr/bin/env python3
"""
Deepfake Detector - Gradio Web Interface
Detect AI-generated audio and video deepfakes
"""
import gradio as gr
import subprocess
import tempfile
import os

def detect_audio(audio_file):
    """Detect audio deepfake"""
    if audio_file is None:
        return "❌ Please upload an audio file", None
    try:
        result = subprocess.run(
            ['python3', 'detector.py', '--input', audio_file, '--type', 'audio'],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout.strip()
        if 'is_deepfake' in output:
            return f"✅ Analysis complete\n{output}", "https://via.placeholder.com/100x100.png?text=Audio"
        return f"⚠️ {output}", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def detect_video(video_file):
    """Detect video deepfake"""
    if video_file is None:
        return "❌ Please upload a video file"
    try:
        result = subprocess.run(
            ['python3', 'detector.py', '--input', video_file, '--type', 'video'],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout.strip()
        return f"✅ Analysis complete\n{output}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Deepfake Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Deepfake Detector")
    gr.Markdown("Detect AI-generated audio and video deepfakes. Protect against voice spoofing and face swap fraud.")
    
    with gr.Tab("🔊 Audio Detection"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Upload Audio File", type="filepath")
                detect_audio_btn = gr.Button("🔍 Detect Audio Deepfake", variant="primary")
            with gr.Column():
                audio_result = gr.Textbox(label="Result", lines=3)
        gr.Markdown("**Supported formats:** WAV, MP3, FLAC")
    
    with gr.Tab("🎭 Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video File")
                detect_video_btn = gr.Button("🔍 Detect Video Deepfake", variant="primary")
            with gr.Column():
                video_result = gr.Textbox(label="Result", lines=3)
        gr.Markdown("**Supported formats:** MP4, MOV, AVI")
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tip**: Higher confidence = more likely real. Scores below 0.5 suggest potential manipulation.")

demo.launch(server_name="0.0.0.0", server_port=7861)
