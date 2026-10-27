# Paddy Crop Disease Detection Using CNN

## Project Overview
This project uses Deep Learning and a Convolutional Neural Network (CNN) to classify paddy crop leaf images into four categories.

## Technologies Used
- Python
- TensorFlow and Keras
- Streamlit
- NumPy
- Pillow

## Disease Categories
- Bacterialblight
- Blast
- Brownspot
- Tungro

## Model Training
The model was trained using a dataset of 5,932 images, split into training and validation sets.

Validation accuracy in the training run: 97.39%.

This result is from one validation split and does not guarantee real-world accuracy.

## How to Run
1. Install the required Python packages.
2. Run `python train_model.py` to train the model.
3. Run `python -m streamlit run app.py` to open the application.

## Disclaimer
This is an experimental educational project. Please consult an agricultural expert to confirm suspected crop diseases.
