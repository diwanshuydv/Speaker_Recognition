import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

def extract_mfcc(parent_dir, sub_folders, n_mfcc=13, max_pad_len=129 , mfcc_window_len= 43):
    x = []
    y = []
    
    for label, folder in enumerate(sub_folders):
        folder_path = os.path.join(parent_dir, folder)
        
        # Loop through each audio file in the speaker's folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):  # Only process .wav files
                file_path = os.path.join(folder_path, file_name)
                
                audio, sr = librosa.load(file_path, sr=None)

                # Extract MFCC features
                org_mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
                delta_mfcc = librosa.feature.delta(org_mfcc)
                delta2_mfcc = librosa.feature.delta(org_mfcc , order=2)
                mfcc=np.concatenate((org_mfcc, delta_mfcc, delta2_mfcc), axis=0)
                
                scaler = StandardScaler()
                mfcc = scaler.fit_transform(mfcc.T)
#                 mfcc.T

                # Padding or truncating the MFCC feature array
                if mfcc.shape[0] < max_pad_len:
                    pad_width = max_pad_len - mfcc.shape[0]
                    mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
                else:
                    mfcc = mfcc[:max_pad_len ,:]

                # Slice the MFCC into windows of window_len
                num_windows = mfcc.shape[0] // mfcc_window_len
                for i in range(num_windows):
                    start = i * mfcc_window_len
                    end = start + mfcc_window_len
                    mfcc_window = mfcc[start:end,: ]
                    x.append(mfcc_window)
                    speaker_id = int(folder[-2:])
                    y.append(speaker_id)
    
    x= np.array(x)
    y= np.array(y)
    return x,y



def speakers_list(no_speakers_file ,data_file ):
    speaker_l = []

    # Get all subfolders in the data_file directory
    subfolders = [f.name for f in os.scandir(data_file) if f.is_dir()]

    # Check if the requested number of speakers is available
    if no_speakers_file > len(subfolders):
        raise ValueError(f"Requested {no_speakers_file} speakers, but only {len(subfolders)} available.")

    # Select the first 'no_speakers_file' subfolders
    speaker_l = subfolders[:no_speakers_file]

    return speaker_l

# speaker_list = speakers_list(no_speakers_file,data_file )
