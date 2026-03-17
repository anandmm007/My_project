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
        
        # Decode base64
        image_bytes = base64.b64decode(image_b64)
        
        return {
            'mime_type': 'image/jpeg',
            'data': image_bytes,
            'data_b64': image_b64,
            'filename': f"webcam_frame_{uuid.uuid4().hex}.jpg",
            'size': len(image_bytes)
        }
    except Exception as e:
        logger.error(f"Error processing webcam frame: {e}")
        raise ValueError("Failed to process webcam frame")

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_available': model is not None,
        'session_stats': session_manager.get_session_stats(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/start_chat', methods=['POST'])
def start_chat():
    """Handle initial webcam frame and first prompt"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_b64 = data.get('image_b64')
        prompt_text = data.get('prompt', '').strip()
        
        if not image_b64:
            return jsonify({'error': 'No webcam frame received'}), 400
        
        if not prompt_text:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if len(prompt_text) > 2000:
            return jsonify({'error': 'Prompt too long (max 2000 characters)'}), 400
        
        # Check model availability
        if not model:
            return jsonify({'error': 'AI model not available. Please try again later.'}), 503
        
        # Process webcam frame
        try:
            image_data = process_webcam_frame(image_b64)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Create session
        session_id = session_manager.create_session(image_data)
        
        # Prepare message with webcam frame
        user_message = {
            'role': 'user',
            'parts': [
                {
                    'inline_data': {
                        'mime_type': image_data['mime_type'],
                        'data': image_data['data_b64']
                    }
                },
                {'text': prompt_text}
            ]
        }
        
        try:
            # Generate response
            response = model.generate_content(
                [user_message],
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            if not response.text:
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    if reason == "SAFETY":
                        return jsonify({'error': 'Content blocked due to safety concerns. Please try different content.'}), 400
                    else:
                        return jsonify({'error': f'Response generation failed: {reason}'}), 500
                else:
                    return jsonify({'error': 'No response generated. Please try again.'}), 500
            
            model_message = {'role': 'model', 'parts': [{'text': response.text}]}
            
            # Update session
            session_manager.update_session_history(session_id, user_message, model_message)
            
            logger.info(f"Started conversation for session {session_id}")
            
            return jsonify({
                'session_id': session_id,
                'response': response.text
            })
            
        except Exception as e:
            logger.error(f"Error generating initial response: {e}")
            return jsonify({'error': 'Failed to generate response. Please try again.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in start_chat: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle subsequent text chat messages"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        session_id = data.get('session_id')
        prompt_text = data.get('prompt', '').strip()
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if not prompt_text:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if len(prompt_text) > 2000:
            return jsonify({'error': 'Message too long (max 2000 characters)'}), 400
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired. Please start a new conversation.'}), 404
        
        # Check model availability
        if not model:
            return jsonify({'error': 'AI model not available. Please try again later.'}), 503
        
        # Rate limiting
        if session.get('message_count', 0) >= 50:
            return jsonify({'error': 'Message limit reached. Please start a new conversation.'}), 429
        
        # Prepare message
        user_message = {'role': 'user', 'parts': [{'text': prompt_text}]}
        
        try:
            # Build conversation history
            conversation_history = []
            
            # Add initial webcam frame context for recent messages
            if session.get('message_count', 0) < 5:
                conversation_history.append({
                    'role': 'user',
                    'parts': [
                        {
                            'inline_data': {
                                'mime_type': session['image']['mime_type'],
                                'data': session['image']['data_b64']
                            }
                        },
                        {'text': '[Webcam frame context for ongoing conversation]'}
                    ]
                })
            
            # Add recent history (last 10 messages)
            recent_history = session['history'][-10:] if len(session['history']) > 10 else session['history']
            conversation_history.extend(recent_history)
            
            # Add current message
            conversation_history.append(user_message)
            
            # Generate response
            response = model.generate_content(
                conversation_history,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            if not response.text:
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    if reason == "SAFETY":
                        return jsonify({'error': 'Message blocked due to safety concerns.'}), 400
                    else:
                        return jsonify({'error': f'Response generation failed: {reason}'}), 500
                else:
                    return jsonify({'error': 'No response generated. Please try again.'}), 500
            
            model_message = {'role': 'model', 'parts': [{'text': response.text}]}
            
            # Update session
            session_manager.update_session_history(session_id, user_message, model_message)
            
            return jsonify({'response': response.text})
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return jsonify({'error': 'Failed to generate response. Please try again.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handle voice input transcription for initial prompt"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if not audio_file or audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        if not model:
            return jsonify({'error': 'AI model not available.'}), 503
        
        try:
            transcription_prompt = "Transcribe the following audio accurately. Only return the transcribed text:"
            
            response = model.generate_content([
                transcription_prompt,
                {
                    "mime_type": audio_file.content_type,
                    "data": base64.b64encode(audio_file.read()).decode()
                }
            ])
            
            if not response.text:
                return jsonify({'error': 'Failed to transcribe audio.'}), 500
            
            transcript = response.text.strip()
            
            if len(transcript) < 2:
                return jsonify({'error': 'Audio too short or unclear.'}), 400
            
            logger.info(f"Successfully transcribed audio: {len(transcript)} characters")
            
            return jsonify({'transcript': transcript})
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return jsonify({'error': 'Failed to process audio.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in transcribe: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

@app.route('/chat_voice', methods=['POST'])
def chat_voice():
    """Handle voice input for ongoing chat"""
    try:
        session_id = request.form.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if not audio_file or audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found or expired.'}), 404
        
        if not model:
            return jsonify({'error': 'AI model not available.'}), 503
        
        # Rate limiting
        if session.get('message_count', 0) >= 50:
            return jsonify({'error': 'Message limit reached.'}), 429
        
        try:
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            if len(audio_bytes) == 0:
                return jsonify({'error': 'Audio file is empty'}), 400
            
            # Transcribe audio
            transcription_response = model.generate_content([
                "Transcribe the following audio accurately. Only return the transcribed text:",
                {
                    "mime_type": audio_file.content_type,
                    "data": audio_b64
                }
            ])
            
            if not transcription_response.text:
                return jsonify({'error': 'Failed to transcribe audio.'}), 500
            
            transcript = transcription_response.text.strip()
            
            if len(transcript) < 2:
                return jsonify({'error': 'Audio too short or unclear.'}), 400
            
            # Process as chat message
            user_message = {'role': 'user', 'parts': [{'text': transcript}]}
            
            # Build conversation history
            conversation_history = []
            
            if session.get('message_count', 0) < 5:
                conversation_history.append({
                    'role': 'user',
                    'parts': [
                        {
                            'inline_data': {
                                'mime_type': session['image']['mime_type'],
                                'data': session['image']['data_b64']
                            }
                        },
                        {'text': '[Webcam frame context]'}
                    ]
                })
            
            recent_history = session['history'][-10:] if len(session['history']) > 10 else session['history']
            conversation_history.extend(recent_history)
            conversation_history.append(user_message)
            
            # Generate response
            response = model.generate_content(
                conversation_history,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            if not response.text:
                return jsonify({'error': 'No response generated.'}), 500
            
            model_message = {'role': 'model', 'parts': [{'text': response.text}]}
            session_manager.update_session_history(session_id, user_message, model_message)
            
            logger.info(f"Processed voice message for session {session_id}")
            
            return jsonify({
                'transcript': transcript,
                'response': response.text
            })
            
        except Exception as e:
            logger.error(f"Error processing voice chat: {e}")
            return jsonify({'error': 'Failed to process voice message.'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in chat_voice: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def payload_too_large_error(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Webcam Chat server on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model available: {model is not None}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
