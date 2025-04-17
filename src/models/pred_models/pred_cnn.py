import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.cnn_complex import SpeakerCNN

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
)


class SpeakerCNN(nn.Module):
    def __init__(self, input_shape, no_speakers, dropout_rate=0.5):
        super(SpeakerCNN, self).__init__()

        self.time_frames, self.mfcc_features = input_shape

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual Block 1
        self.res1_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(64)
        self.res1_shortcut = nn.Conv2d(32, 64, kernel_size=1)

        # Residual Block 2
        self.res2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_bn2 = nn.BatchNorm2d(128)
        self.res2_shortcut = nn.Conv2d(64, 128, kernel_size=1)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, no_speakers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional Block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Block 1
        shortcut = self.res1_shortcut(x)
        x = F.relu(self.res1_bn1(self.res1_conv1(x)))
        x = self.res1_bn2(self.res1_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Residual Block 2
        shortcut = self.res2_shortcut(x)
        x = F.relu(self.res2_bn1(self.res2_conv1(x)))
        x = self.res2_bn2(self.res2_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def pred_speaker(wav_path: str, max_pad_len: int = 129):
    audio, sr = librosa.load(wav_path, sr=None)

    # Extract MFCC features
    org_mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(org_mfcc)
    delta2_mfcc = librosa.feature.delta(org_mfcc, order=2)
    mfcc = np.concatenate((org_mfcc, delta_mfcc, delta2_mfcc), axis=0)

    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc.T)
    if mfcc.shape[0] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_pad_len, :]
    x_test_tensor = torch.tensor(mfcc, dtype=torch.float32)
    model = SpeakerCNN(input_shape=mfcc.shape, no_speakers=51, dropout_rate=0.2867160906483275)
    model.load_state_dict(torch.load("../models/final_model_cnn_30_3_3.pth"))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_tensor.unsqueeze(0).to(device))
        pred = torch.argmax(outputs, axis=1).cpu().item()

    return pred


# print("pred--", pred_speaker("/home/raid3/Diwanshu/prml/prml_speaker_recog_spring_2025/data/processed3/50_speakers_audio_data/Speaker_0003/Speaker_0003_Speaker_0003_00000_chunk2.wav"))
