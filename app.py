import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load preprocessed data
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Define the ANN model
model = keras.Sequential([
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(1)  # Regression output (no activation function)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[early_stopping], verbose=1)

# Save the trained model
model.save("tf_bridge_model.h5")

# Plot training & validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
