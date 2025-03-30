import os
import librosa
import soundfile as sf

def create_audio_chunks(input_dir, output_dir, chunk_duration=5):
    """
    Splits all audio files in each speaker's folder into 5-second non-overlapping chunks.

    Args:
        input_dir (str): Path to the directory containing speaker subdirectories with audio files.
        output_dir (str): Path to save the processed audio chunks.
        chunk_duration (int): Duration of each chunk in seconds (default: 5).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist
    
    # Iterate through each speaker folder inside input directory
    for speaker in os.listdir(input_dir):
        speaker_path = os.path.join(input_dir, speaker)  # Full path to speaker folder
        
        if not os.path.isdir(speaker_path):
            continue  # Skip if not a directory (to avoid non-folder files)
        
        speaker_output_path = os.path.join(output_dir, speaker)
        os.makedirs(speaker_output_path, exist_ok=True)  # Ensure speaker output folder exists
        
        # Iterate through all .wav files inside each speaker's folder
        for file_idx, audio_file in enumerate(os.listdir(speaker_path)):
            if not audio_file.endswith(".wav"):  # Process only WAV files
                continue
            
            file_path = os.path.join(speaker_path, audio_file)
            y, sr = librosa.load(file_path, sr=None)  # Load audio with original sampling rate
            chunk_samples = chunk_duration * sr  # Convert chunk duration to samples
            total_samples = len(y)  # Total samples in the audio file
            
            # Split audio into non-overlapping 5-second chunks
            for chunk_idx, start_sample in enumerate(range(0, total_samples, chunk_samples)):
                end_sample = start_sample + chunk_samples
                
                if end_sample > total_samples:  # Skip if the last chunk is shorter than 5 seconds
                    continue
                
                chunk = y[start_sample:end_sample]  # Extract chunk from audio
                chunk_filename = f"{speaker}_file{file_idx}_chunk{chunk_idx}.wav"
                chunk_path = os.path.join(speaker_output_path, chunk_filename)
                
                # Save the chunk as a new audio file
                sf.write(chunk_path, chunk, sr)
                print(f"Saved: {chunk_path}")
    
    print("Processing complete!")

# Example usage
input_directory = "data/raw/50_speakers_audio_data"
output_directory = "data/processed/50_speakers_audio_data"
create_audio_chunks(input_directory, output_directory)
