 Speech Emotion Recognition Using CNN and MFCCs

##  Project Overview

This project, **Speech Emotion Recognition Using CNN and MFCCs**, aims to identify human emotions from speech audio recordings using Machine Learning and Deep Learning techniques.

The project uses Python to process audio files, extract important speech features, and train a Convolutional Neural Network (CNN) to classify different emotions.

This project was developed as part of the **CodeAlpha Machine Learning Internship**.

##  Objectives

* Recognize emotions from speech audio recordings.
* Extract meaningful features from audio using MFCCs.
* Train a CNN model for emotion classification.
* Evaluate the model's performance on speech data.
* Gain practical experience in audio processing and Deep Learning.

##  Technologies Used

* **Python** — Programming language
* **Librosa** — Audio processing and feature extraction
* **NumPy** — Numerical operations
* **TensorFlow and Keras** — Building and training the CNN model
* **Scikit-learn** — Data preprocessing, splitting, and model evaluation
* **VS Code** — Development environment
* **RAVDESS** — Speech emotion dataset

##  Dataset

This project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

The dataset contains audio recordings expressing different emotions.

**Dataset link:** https://www.kaggle.com/dsv/256618

The speech dataset includes eight emotion categories:

| Code | Emotion   |
| ---- | --------- |
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

The project processes the speech audio files and extracts features for model training.

##  Project Workflow

1. **Load the dataset:** Read speech audio files from the RAVDESS dataset.
2. **Preprocess audio:** Load and prepare audio recordings for feature extraction.
3. **Extract MFCC features:** Convert speech audio into numerical features using Mel-Frequency Cepstral Coefficients (MFCCs).
4. **Prepare the data:** Organize the extracted features and emotion labels for training.
5. **Build the CNN model:** Create a Convolutional Neural Network using TensorFlow and Keras.
6. **Train the model:** Train the model using the prepared speech features.
7. **Evaluate the model:** Measure performance using accuracy and classification metrics.
8. **Predict emotions:** Use the trained model to predict the emotion associated with speech audio.

##  What Are MFCCs?

**Mel-Frequency Cepstral Coefficients (MFCCs)** are audio features that represent important characteristics of speec
