import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import eventlet
from flask_socketio import SocketIO, emit
from scipy.io import wavfile
import soundfile as sf
import numpy as np
import io
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pred_models.pred_cnn import pred_speaker

app =  Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*",async_mode='eventlet')

from resemblyzer import preprocess_wav, VoiceEncoder
from qdrant_client import QdrantClient
from collections import Counter


# def pred_speaker(wav_path: str, collection_name: str = "speaker_recognition_testing",url: str = "http://localhost:6333"): 
#     client = QdrantClient(url=url)
#     encoder = VoiceEncoder("cuda")
#     test_wav = preprocess_wav(wav_path)
#     test_embeddings = encoder.embed_utterance(test_wav)
#     results = client.search(collection_name, test_embeddings, score_threshold=0.6)
#     top_5_speaker_ids = [result.payload["speaker_id"] for result in results]
#     speaker_id_counts = Counter(top_5_speaker_ids)
#     most_frequent_speaker_id = speaker_id_counts.most_common(1)[0][0]
#     return most_frequent_speaker_id


@app.post('/api/upload')
def upload():
    print("Received request")
    if request.method == 'POST':
        file = request.files['file']
        file.save("./temp/test_audio.wav")
        print(file)
        result = pred_speaker("./temp/test_audio.wav")
        return jsonify({'result': result})
    return jsonify({'error': 'Invalid request'}), 400

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
    
    try:
        # Use soundfile to decode and resample
        audio_np, samplerate = sf.read(io.BytesIO(data))
        print(f"Original sample rate: {samplerate}, shape: {audio_np.shape}")

        # Step 2: Resample to 22050 Hz
        if samplerate != 22050:
            audio_np = sf.resample(audio_np, samplerate, 22050)
            samplerate = 22050

        # Step 3: Convert to int16 format (standard for wav)
        audio_int16 = np.int16(audio_np * 32767)

        # Step 4: Save using scipy.io.wavfile
        wavfile.write("./temp/test_audio.wav", samplerate, audio_int16)

        print("Audio saved successfully as test_audio.wav")

    except Exception as e:
        print("Error processing audio:", e)
    result = pred_speaker("./temp/test_audio.wav")
    print(result)
    emit('recognised_speaker', {'data': result})
socketio.run(app, port=3000, debug=True)