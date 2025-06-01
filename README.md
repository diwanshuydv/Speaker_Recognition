# CSL2060_speaker_recognition_spring_2025
If you encounter the error:
```sh
'vite' is not recognized as an internal or external command
```
Follow these steps to resolve it:

1.Ensure dependencies are installed
Run the following command in the project root:
```sh
npm install
```

2.Manually install Vite (if needed)
```sh
npm install vite --save-dev
```

3.Verify the `package.json` script section contains:
```json
"scripts": {
"dev": "vite",
"build": "vite build",
"serve": "vite preview"
}
```

4.Delete `node_modules` and reinstall dependencies (if issues persist)
```sh
rm -rf node_modules package-lock.json
npm install
npm run dev
```
(For Windows PowerShell: use `Remove-Item -Recurse -Force node_modules, package-lock.json` instead of `rm -rf`.)

### Installing Librosa  

5. Librosa can be installed using pip with the following command:  
```bash
pip install librosa
#Ensure you have the necessary dependencies by running:

pip install numpy scipy joblib soundfile audioread  



6. Technologies Used
Frontend: React.js, Vite

Backend: Python, Flask

Audio Processing: Librosa, NumPy, SciPy

Machine Learning: SVM (Support Vector Machine)

7. Future Improvements

🎯 Incorporate deep learning models for improved accuracy

📈 Add visual plots of MFCC features

🗂️ Improve multi-speaker classification support

🛡️ Add authentication and user profiles for voice registration

8. License

This project is for academic use only. Please contact the authors for reuse.


