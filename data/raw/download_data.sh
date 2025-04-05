#!/bin/bash

# Exit on error
set -e

echo "Downloading speaker recognition dataset from Kaggle..."

# Step 1: Download the dataset from Kaggle
wget https://www.kaggle.com/api/v1/datasets/download/vjcalling/speaker-recognition-audio-dataset

# Step 2: Rename the downloaded file to a .zip file
mv speaker-recognition-audio-dataset speaker-recognition-audio-dataset.zip

# Step 3: Unzip the downloaded dataset
unzip speaker-recognition-audio-dataset.zip

# Step 4: Remove the zip file to save space
rm speaker-recognition-audio-dataset.zip

echo "Download, extraction, and cleanup complete!!!"