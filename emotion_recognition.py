import os
import glob
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Emotion names in the RAVDESS dataset
emotions = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


# Extract MFCC features from audio
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc, axis=1)


# Load audio files
X = []
y = []

audio_files = glob.glob(
    os.path.join("Actor_*", "*.wav")
)

print("Loading audio files...")

for file in audio_files:
    try:
        filename = os.path.basename(file)
        parts = filename.split("-")

        # Use speech recordings only
        if parts[0] != "03" or parts[1] != "01":
            continue

        emotion = emotions.get(parts[2])

        if emotion is None:
            continue

        features = extract_features(file)

        X.append(features)
        y.append(emotion)

    except Exception as error:
        print("Skipped file:", file)


# Check the dataset
if len(X) < 10:
    print("Not enough audio files found.")
    print("Check your dataset folders.")
    raise SystemExit


X = np.array(X)
y = np.array(y)

print("Total audio files:", len(X))


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training the model...")
model.fit(X_train, y_train)


# Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")


# Predict the emotion of one audio file
sample_file = "Actor_01/03-01-04-01-01-01-01.wav"

sample_features = extract_features(sample_file)

sample_features = sample_features.reshape(1, -1)

predicted_emotion = model.predict(sample_features)

print("\nSample audio:", sample_file)
print("Predicted Emotion:", predicted_emotion[0])