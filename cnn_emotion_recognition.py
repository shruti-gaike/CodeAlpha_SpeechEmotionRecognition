import os
import glob
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# 1. Emotion labels in the RAVDESS dataset
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


# 2. Extract MFCC features from audio
def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        duration=3
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )

    # Make every MFCC feature array the same size
    max_length = 130

    if mfcc.shape[1] < max_length:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, max_length - mfcc.shape[1]))
        )
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc


# 3. Load the audio dataset
X = []
y = []

audio_files = glob.glob(
    os.path.join("Actor_*", "*.wav")
)

print("Loading audio files...")

for file_path in audio_files:

    try:
        filename = os.path.basename(file_path)
        parts = filename.split("-")

        # Select speech recordings only
        if parts[0] != "03" or parts[1] != "01":
            continue

        emotion = emotions.get(parts[2])

        if emotion is None:
            continue

        features = extract_features(file_path)

        X.append(features)
        y.append(emotion)

    except Exception as error:
        print("Skipping file:", file_path, error)


# 4. Check the dataset
if len(X) == 0:
    raise ValueError(
        "No audio files found. Check your dataset folders."
    )

X = np.array(X, dtype=np.float32)

# Normalize MFCC features
mean = np.mean(X, axis=(0, 2), keepdims=True)
std = np.std(X, axis=(0, 2), keepdims=True)

X = (X - mean) / (std + 1e-6)
y = np.array(y)

print("Total audio files:", len(X))
print("Feature shape:", X.shape)


# 5. Convert emotion names into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# CNN expects: samples, height, width, channels
X = X[..., np.newaxis]


# 6. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# 7. Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(40, 130, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(
        64, (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(
        128, (3, 3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])



# 8. Compile the CNN
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 9. Display the model structure
model.summary()


# 10. Train the CNN model
print("\nTraining the CNN model...")
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# 13. Predict one audio recording
sample_file = glob.glob(
    os.path.join("Actor_*", "*.wav")
)[0]

sample_features = extract_features(sample_file)

sample_features = sample_features[
    np.newaxis, ..., np.newaxis
]

sample_prediction = model.predict(
    sample_features,
    verbose=0
)

predicted_index = np.argmax(sample_prediction)

predicted_emotion = encoder.inverse_transform(
    [predicted_index]
)[0]

print("\nSample Audio:", sample_file)
print("Predicted Emotion:", predicted_emotion)


# 14. Save the trained model
model.save("cnn_emotion_model.keras")

print("\nModel saved as cnn_emotion_model.keras")