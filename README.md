# Gender Classification System

This project implements a **Gender Classification System** using Deep Learning and a Flask-based web interface.  
It allows users to upload an image and predicts the gender using a trained Convolutional Neural Network (CNN).

---

## 📂 Project Structure

```
Gender_Classi_Grand_Final/
│
├── Gender_Classi_02/
│   ├── app.py                 # Flask web application entry point
│   ├── model.py               # Model architecture and loading
│   ├── train.py               # Training script
│   ├── test.py                # Testing script
│   │
│   ├── Model/
│   │   ├── Gcuf_1000.h5       # Pre-trained model file
│   │   └── GenderClassifyMF.h5 # Another trained model
│   │
│   ├── static/                # CSS, JavaScript, and image files
│   │   ├── css/               # Bootstrap, FontAwesome, and custom styles
│   │   ├── js/                # jQuery, Bootstrap JS, and custom scripts
│   │   └── images             # Sample images
│   │
│   ├── templates/             # HTML templates for the web app
│   │   ├── index.html
│   │   └── import.html
│   │
│   └── upload/                # Uploaded images for testing
│
└── README.md (this file)
```

---

## 🚀 How to Run

1. Clone or extract the repository.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```
5. Upload an image to classify gender.

---

## 🧠 Model Information
- The project uses **Convolutional Neural Networks (CNNs)** for gender classification.
- Trained models are stored in the `Model/` directory (`.h5` format).
- The model predicts **Male** or **Female** from images.

---

## 📊 Files Description
- **app.py** → Main Flask web app
- **model.py** → Model definition and loading
- **train.py** → Script to train the CNN
- **test.py** → Script to test the trained model
- **Model/** → Pre-trained model weights
- **static/** → CSS, JS, images for frontend
- **templates/** → HTML pages
- **upload/** → Sample uploaded images

---

## ✨ Features
- Web-based interface for uploading and predicting gender.
- Pre-trained deep learning models included.
- Bootstrap-based responsive frontend.
- Easy to extend and retrain with new datasets.

---

## 📌 Requirements
- Python 3.x
- TensorFlow / Keras
- Flask
- NumPy, OpenCV, Matplotlib (for training & testing)

Install dependencies:
```bash
pip install flask tensorflow keras numpy opencv-python matplotlib
```

---

## 👨‍💻 Author
Developed as a **Gender Classification System Project** using CNN and Flask.

