#pip install -r requirements.txt
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import base64
import uuid
import logging
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, Optional
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
# Configure Gemini API
api_key = "AIzaSyAj7QePSFe7l4dVBBcfbLD8cYXt-uc2bH0"
if not api_key:
    logger.error("GOOGLE_API_KEY environment variable not set")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=api_key)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Enhanced session storage with automatic cleanup
class SessionManager:
    def __init__(self, cleanup_interval: int = 300):
        self.sessions: Dict[str, dict] = {}
        self.cleanup_interval = cleanup_interval
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Start background thread for session cleanup"""
        def cleanup_expired_sessions():
            while True:
                try:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session_data in self.sessions.items():
                        if current_time - session_data['last_activity'] > timedelta(hours=1):
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        logger.info(f"Cleaned up expired session: {session_id}")
                    
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Error in session cleanup: {e}")
                    time.sleep(self.cleanup_interval)
        
        cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
        cleanup_thread.start()
    
    def create_session(self, image_data: dict) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'image': image_data,
            'history': [],
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'message_count': 0
        }
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        session = self.sessions.get(session_id)
        if session:
            session['last_activity'] = datetime.now()
        return session
    
    def update_session_history(self, session_id: str, user_message: dict, model_response: dict):
        """Update session history"""
        if session_id in self.sessions:
            self.sessions[session_id]['history'].extend([user_message, model_response])
            self.sessions[session_id]['last_activity'] = datetime.now()
            self.sessions[session_id]['message_count'] += 1
    
    def get_session_stats(self) -> dict:
        """Get session statistics"""
        return {
            'active_sessions': len(self.sessions),
            'total_messages': sum(s['message_count'] for s in self.sessions.values())
        }
# Initialize session manager
session_manager = SessionManager()
# Initialize Gemini models with error handling
def get_gemini_model():
    """Get Gemini model with fallback options"""
    models_to_try = [
        'gemini-2.0-flash-exp',
        'gemini-1.5-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-pro',
                'gemini-1.5-pro-latest',
        'gemini-pro-vision'
    ]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            logger.info(f"Using Gemini model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            continue
    
    raise Exception("No Gemini models available")
# Initialize model
try:
    model = get_gemini_model()
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    model = None
# Enhanced safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]
# Generation configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}
def process_webcam_frame(image_b64: str) -> dict:
    """Process webcam frame data"""
try:
# Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
