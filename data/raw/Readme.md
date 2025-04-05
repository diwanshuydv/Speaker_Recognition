# 📥 Speaker Recognition Dataset Downloader

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Shell Script](https://img.shields.io/badge/script-bash-1f425f.svg)
![Data Source: Kaggle](https://img.shields.io/badge/data-Kaggle-blue)

This script automates the process of downloading and extracting the **Speaker Recognition Audio Dataset** from Kaggle for use in this speaker identification project.

---

## 📄 Overview

The `download_data.sh` script performs the following:

- 📦 Downloads the dataset using `wget` 

- 📂 Unzips the dataset in the current directory

---

## ⚙️ Requirements

- `wget`
- `unzip`

---

## 🚀 Usage

1. Make the script executable:
   ```bash
   chmod +x download_data.sh
   ```

2. Run it:
   ```bash
   ./download_data.sh
   ```

---

## 📁 Output

- `speaker-recognition-audio-dataset.zip` – Downloaded dataset archive
- Extracted dataset folder with all audio files and metadata

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🧠 Acknowledgements

- Speaker Recognition Audio Dataset on Kaggle
