import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerCNN(nn.Module):
    def __init__(self, input_shape, no_speakers):
        super(SpeakerCNN, self).__init__()
        
        # Unpack input shape dimensions
        self.time_frames, self.mfcc_features = input_shape  # Example: input_shape=(29, 39)

        # First convolution block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # Second convolution block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # Third convolution block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Calculate the output shape after 3 pooling layers
        conv_out_time = self._calculate_output_dim(self.time_frames, 3)
        conv_out_features = self._calculate_output_dim(self.mfcc_features, 3)
        flattened_dim = 128 * conv_out_time * conv_out_features

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_dim, 256)
        self.fc2 = nn.Linear(256, no_speakers)

    def _calculate_output_dim(self, size, num_pools):
        # Calculates the output dimension after a series of MaxPool2d(2)
        for _ in range(num_pools):
            size = size // 2
        return size

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



        