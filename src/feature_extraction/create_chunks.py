import os
import librosa
import soundfile as sf
import numpy as np

def create_audio_chunks(input_dir, output_dir, chunk_duration=5):
    """
    Processes audio files in speaker directories, creates 5-second chunks,
    and saves them under a structured output directory.

    Args:
        input_dir (str): Path to the input directory containing speaker folders.
        output_dir (str): Path to the output directory for saving chunks.
        chunk_duration (int): Duration of each chunk in seconds (default is 5).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each speaker's directory
    for speaker_dir in os.listdir(input_dir):
        speaker_path = os.path.join(input_dir, speaker_dir)

        # Skip if not a directory
        if not os.path.isdir(speaker_path):
            continue

        # Create corresponding speaker directory in output
        processed_speaker_dir = os.path.join(output_dir, speaker_dir)
        os.makedirs(processed_speaker_dir, exist_ok=True)

        # Process each WAV file in the speaker's directory
        for file_id, file_name in enumerate(os.listdir(speaker_path)):
            if file_name.endswith(".wav"):
                file_path = os.path.join(speaker_path, file_name)
                
                # Load the audio file
                y, sr = librosa.load(file_path, sr=None)  # Native sampling rate

                # Calculate the number of samples per chunk
                chunk_samples = chunk_duration * sr
                total_samples = len(y)

                # Split into chunks
                for chunk_id, start_sample in enumerate(range(0, total_samples, chunk_samples)):
                    # Extract the current chunk
                    chunk = y[start_sample:start_sample + chunk_samples]

                    # Skip chunks smaller than the required duration
                    if len(chunk) < chunk_samples:
                        continue

                    # Define the output file path
                    chunk_file_name = f"{speaker_dir}_file{file_id}_chunk{chunk_id}.wav"
                    chunk_file_path = os.path.join(processed_speaker_dir, chunk_file_name)

                    # Save the chunk using soundfile
                    sf.write(chunk_file_path, chunk, sr)

    print(f"Chunks created and saved under '{output_dir}'")

# Example usage
input_directory = "data/raw/50_speakers_audio_data"
output_directory = "/data/processed/50_speakers_audio_data"
create_audio_chunks(input_directory, output_directory, chunk_duration=5)