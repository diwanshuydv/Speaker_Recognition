from pathlib import Path
from typing import Union, List
from torch import nn
from time import perf_counter as timer
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


class VoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device] = None, verbose=True, weights_fpath: Union[Path, str] = None):
        """
        If None, defaults to cuda if available, otherwise runs on cpu. Embeddings are always returned
        on the cpu as numpy arrays.
        :param weights_fpath: path to a pretrained weights file (e.g. "pretrained.pt")
        """
        super().__init__()

        # Define the CNN network.
        # This network treats the input mel spectrogram as a 2D image (adding a channel dim).
        # It consists of several convolutional blocks and an adaptive pooling layer.
        self.cnn = nn.Sequential(
            # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: Increase depth to capture more nuanced features
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Use adaptive average pooling to produce a fixed size output irrespective of input dimensions
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Fully connected layer to map from CNN output to embedding dimension.
        self.fc = nn.Linear(256, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model weights if provided.
        if weights_fpath is None:
            weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        else:
            weights_fpath = Path(weights_fpath)

        if not weights_fpath.exists():
            raise Exception("Couldn't find the voice encoder pretrained model at %s." %
                             weights_fpath)
        start = timer()
        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        if verbose:
            print("Loaded the voice encoder model on %s in %.2f seconds." %
                  (device.type, timer() - start))

    def forward(self, mels: torch.FloatTensor):
        """
        Computes embeddings for a batch of utterance spectrograms.
        :param mels: float32 tensor of shape (batch_size, n_frames, mel_n_channels)
        :return: L2-normed embeddings of shape (batch_size, model_embedding_size)
        """
        # Add a channel dimension: expecting shape (batch_size, 1, n_frames, mel_n_channels)
        mels = mels.unsqueeze(1)
        # Forward through CNN
        x = self.cnn(mels)
        assert not torch.isnan(x).any(), "NaN in CNN"  # shape: (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)
        embeds_raw = self.relu(self.fc(x))
        assert not torch.isnan(embeds_raw).any(), "NaN in ReLU"
        # Normalize the embeddings to lie in the range [0, 1]
        print("here")
        eps = 1e-7  # Small value to avoid division by zero
        norms = torch.norm(embeds_raw, dim=1, keepdim=True).clamp(min=eps)
        normalized_embeds = embeds_raw / norms
        return normalized_embeds

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        """
        Computes slices to divide an utterance waveform into partial utterances.
        Returns waveform slices and corresponding mel spectrogram slices.
        """
        assert 0 < min_coverage <= 1

        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= partials_n_frames, "The rate is too low; it should be at least %f" % \
            (sampling_rate / (samples_per_frame * partials_n_frames))

        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75):
        """
        Computes an embedding for a single utterance by splitting it into partials and averaging.
        :param wav: preprocessed waveform (numpy array, float32)
        :param return_partials: if True, returns partial embeddings and corresponding waveform slices
        :param rate: partial utterance rate per second
        :param min_coverage: minimum coverage required for the final partial
        :return: embedding (and optionally partial embeddings & slices)
        """
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        mel = wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels).cpu().numpy()

        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)
        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        """
        Computes a speaker embedding by averaging the embeddings of multiple utterances.
        :param wavs: list of wavs (numpy arrays, float32)
        :return: L2-normalized speaker embedding
        """
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs)
                             for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)


### Training Loop Function

def train_voice_encoder(model, dataloader, num_epochs, optimizer, criterion, device, save_path):
    """
    Trains the voice encoder model using a triplet loss.

    :param model: instance of VoiceEncoder
    :param dataloader: DataLoader yielding triplets in the form (anchor, positive, negative) where
                       each is a batch of mel spectrograms as a float32 tensor
    :param num_epochs: number of training epochs
    :param optimizer: optimizer (e.g., Adam or SGD)
    :param criterion: loss function (e.g., nn.TripletMarginLoss)
    :param device: torch.device on which to run training
    """
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            # Assume batch is a tuple of (anchor, positive, negative)
            anchor, positive, negative = batch
            print(f"Anchor shape: {anchor.shape}, Positive shape: {positive.shape}, Negative shape: {negative.shape}")
            print("Anchor mean:", anchor.mean().item(), "std:", anchor.std().item())
            print("Positive mean:", positive.mean().item(), "std:", positive.std().item())
            print("Negative mean:", negative.mean().item(), "std:", negative.std().item())
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Forward pass: compute embeddings for each sample in the triplet
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            assert not torch.isnan(emb_anchor).any(), "Model output contains NaN"
            assert not torch.isnan(emb_negative).any(), "Model output contains NaN"
            assert not torch.isnan(emb_positive).any(), "Model output contains NaN"
            # Compute the triplet loss
            loss = criterion(emb_anchor + 1e-8, emb_positive + 1e-8, emb_negative + 1e-8)
            if torch.isnan(loss):
                print("Loss is NaN. Stopping training.")
                break
            print(f"Loss: {loss.item()}")
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    # Save the final model
    torch.save({"model_state": model.state_dict()}, save_path)
    print(f"Model saved to {save_path}")


class TripletDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X: numpy array or list of feature vectors (e.g., mel-spectrograms).
            y: list or numpy array of corresponding labels.
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor = self.X[idx]
        positive = self.X[random.choice(range(len(self.X)))]
        negative = self.X[random.choice(range(len(self.X)))]
        return anchor, positive, negative


# Example usage:
# Define your model, optimizer, loss criterion, etc.
model = VoiceEncoder(device="cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.TripletMarginLoss(margin=1.0)
train_dataset = TripletDataset(X_train, y_train)  # Replace X_train and y_train with your dataset
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the model
train_voice_encoder(model, train_dataloader, num_epochs=10, optimizer=optimizer, criterion=criterion, device="cuda", save_path="voice_encoder.pth")
