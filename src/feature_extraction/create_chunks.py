import os
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_audio_file(speaker, audio_file, speaker_path, speaker_output_path, chunk_duration):
    """
    Processes a single audio file by chunking and saving to disk.
    """
    if not audio_file.endswith(".wav"):
        return

    file_path = os.path.join(speaker_path, audio_file)
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    
    chunk_samples = chunk_duration * sr
    total_samples = len(y)
    
    for chunk_idx, start_sample in enumerate(range(0, total_samples, chunk_samples)):
        end_sample = start_sample + chunk_samples
        if end_sample > total_samples:
            continue

        chunk = y[start_sample:end_sample]
        chunk_filename = f"{speaker}_{os.path.splitext(audio_file)[0]}_chunk{chunk_idx}.wav"
        chunk_path = os.path.join(speaker_output_path, chunk_filename)
        try:
            sf.write(chunk_path, chunk, sr)
            print(f"Saved: {chunk_path}")
        except Exception as e:
            print(f"Error saving {chunk_path}: {e}")

def create_audio_chunks(input_dir, output_dir, chunk_duration=2, max_threads=None):
    """
    Splits all audio files in each speaker's folder into chunks using multiple threads.
    Uses a safe number of threads based on CPU count.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if max_threads is None:
        max_threads = os.cpu_count() * 2  # Safe for I/O-bound tasks

    futures = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for speaker in os.listdir(input_dir):
            speaker_path = os.path.join(input_dir, speaker)
            if not os.path.isdir(speaker_path):
                continue

            speaker_output_path = os.path.join(output_dir, speaker)
            os.makedirs(speaker_output_path, exist_ok=True)

            for audio_file in os.listdir(speaker_path):
                future = executor.submit(
                    process_audio_file,
                    speaker,
                    audio_file,
                    speaker_path,
                    speaker_output_path,
                    chunk_duration
                )
                futures.append(future)
        
        for future in as_completed(futures):
            future.result()

    print("Processing complete!")

# Example usage
input_directory = "data/raw/50_speakers_audio_data"
output_directory = "data/processed33/50_speakers_audio_data"
create_audio_chunks(input_directory, output_directory)