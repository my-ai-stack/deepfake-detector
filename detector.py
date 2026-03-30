#!/usr/bin/env python3
"""Deepfake Detector - Using Ollama for AI analysis"""
import sys
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder"

def detect_audio(file_path: str) -> dict:
    prompt = f"Analyze this audio file for signs of being AI-generated or deepfake. File: {file_path}"
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            analysis = response.json().get("response", "")
            return {"is_deepfake": None, "confidence": 0, "analysis": analysis[:300]}
    except Exception as e:
        return {"is_deepfake": None, "error": str(e)}
    return {"is_deepfake": None, "error": "Ollama not running"}

if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    result = detect_audio(file)
    print(f"🔍 Analysis: {result}")
