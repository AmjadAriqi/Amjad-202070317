import cv2
import numpy as np
import os
import glob
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Function to load and preprocess images
def load_data(path):
    images = []
    labels = []

    # Load ripe tomatoes images
    for file in glob.glob(os.path.join(path, 'Images', 'Riped tomato_*.jpeg')):
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(1)  # label 1 for ripe tomatoes

    # Load unripe tomatoes images
    for file in glob.glob(os.path.join(path, 'Images', 'unriped tomato_*.jpeg')):
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(0)  # label 0 for unripe tomatoes

    # Convert to numpy arrays and normalize images to 0-1 range
    images = np.array(images) / 255.0
    labels = np.array(labels)

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    return train_data, test_data, train_labels, test_labels

# Function to create the model
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Function to train the model with early stopping, learning rate reduction, and model checkpointing
def train_model(model, train_data, train_labels, test_data, test_labels):
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Reduce learning rate if validation loss stops improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    
    # Save the best model based on validation accuracy
    model_checkpoint = ModelCheckpoint('best_tomato_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    
    # Train the model
    history = model.fit(
        train_data, train_labels, 
        epochs=50, 
        validation_data=(test_data, test_labels),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=2  # To show progress during training
    )
    
    return model

# Function to evaluate the model
def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Load data
train_data, test_data, train_labels, test_labels = load_data('C:\\Users\\ADVANCED\\OneDrive\\Desktop\\Tomato Ripness Detection\\Tomato_Ripeness_detection\\dataset')
# Print the number of images
print(f"Total images: {len(train_data) + len(test_data)}")
print(f"Training images: {len(train_data)}")
print(f"Testing images: {len(test_data)}")

# Create model
model = create_model()

# Train model
model = train_model(model, train_data, train_labels, test_data, test_labels)

# Evaluate model
evaluate_model(model, test_data, test_labels)

