from flask import Flask, request, jsonify
from flask_cors import CORS
import eventlet
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.io.wavfile import write
import wave
app =  Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*",async_mode='eventlet')

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('response', {'data': 'Connected to server!'})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")
    emit('response', {'data': 'Disconnected from server!'})

@socketio.on('audio')
def handle_audio(data:bytes):
    print("Received audio data")
    
    file_path = "audio.wav"
    print(len(data.__bytes__()))
    with open(file_path, 'wb') as wf:
        wf.write(data)
    # with open(file_path, 'wb') as wf:
    #     wf.write(data.__bytes__())
    # eventlet.sleep(0.1)
    # Here you would process the audio data
    # For now, just echo it back to the client
    emit('recognised_speaker', {'data': 'Speaker recognized!'})

socketio.run(app, port=3000, debug=True)