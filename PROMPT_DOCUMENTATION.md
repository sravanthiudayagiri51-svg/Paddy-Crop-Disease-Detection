# AI Skills & Prompt Engineering Documentation

## Project Title
Paddy Crop Disease Detection Web Application

## 1. Objective
The objective of this project is to build a web application that classifies paddy leaf images into four categories using a Convolutional Neural Network (CNN).

## 2. AI Tools Used
- ChatGPT: Project planning, code generation assistance, debugging guidance, and documentation.
- Python 3.11.9: Programming language.
- TensorFlow/Keras: CNN model training and prediction.
- Streamlit: Web application interface.
- VS Code: Code editing and project development.

## 3. Dataset
The dataset contains 5,932 files organized into four classes:
- Bacterialblight: 1,584
- Blast: 1,440
- Brownspot: 1,600
- Tungro: 1,308

## 4. Prompt Engineering Process

### Prompt 1: Project Planning
Prompt summary:
"Help me develop my Paddy Crop Disease Detection project using Python, AI/ML, and a web application for my AI assessment."

Purpose:
To align the project with the assessment requirements.

### Prompt 2: Environment Setup
Prompt summary:
"Guide me step by step to check my Python version, install required packages, and set up the project in VS Code."

Purpose:
To prepare the local development environment.

### Prompt 3: Dataset and Model
Prompt summary:
"Help me build a CNN model using my four paddy disease image classes, train it, and save the trained model."

Purpose:
To train a model using the paddy dataset.

### Prompt 4: Troubleshooting
Prompt summary:
"My training script did not run. Help me identify the problem and run the training successfully."

Purpose:
To identify that the Python file was empty, add the code, and rerun training.

### Prompt 5: Web Application
Prompt summary:
"Create a Streamlit application that lets users upload a paddy leaf image and displays the CNN model's predicted class and scores."

Purpose:
To connect the trained model to a user interface.

## 5. Significant Revisions
- Inspected the old project code and found it used pepper, potato, and tomato classes.
- Changed the approach to train a separate CNN model using the four paddy disease classes.
- Checked the empty training file and saved the training code before running it again.
- Added image upload, prediction scores, and an experimental-use disclaimer to the web app.

## 6. Model Training Results
- Model: Custom CNN
- Image size: 128 × 128
- Training epochs: 5
- Training accuracy at final epoch: 94.42%
- Validation accuracy: 97.39%
- Validation loss: 0.0853

These results are from one training run and do not guarantee performance on new images.

## 7. Sample Prediction
Uploaded image: Brownspot sample image
Predicted class: Brownspot
Displayed model score: 99.88%

This is one sample prediction, not proof of general accuracy.

## 8. Challenges
- Finding the existing project files.
- Identifying the mismatch between the old model classes and the paddy dataset.
- Fixing the empty training script.
- Running model training on a CPU.
- Connecting the trained model to the Streamlit application.

## 9. Outcome
A local Streamlit application was created. It accepts a paddy leaf image and displays a predicted class and model scores.

## 10. Limitations
The model is an experimental prototype. Its predictions may be wrong, and the displayed score is not a verified measure of real-world disease accuracy. Agricultural treatment decisions should be checked with an expert.

## 11. How to Run
Run the following commands from the project folder:

python train_model.py

python -m streamlit run app.py

## 12. AI Usage Statement
ChatGPT was used to assist with planning, code drafting, troubleshooting guidance, and documentation. The code was run locally, and the model training and sample application output were observed on the user's computer.